"""Ascend kernels for the compressor backward.

Vector cores scatter the Hadamard and softmax gradients back onto tokens and
into a private APE partial. Cube cores then run the four projection GEMMs:
``dX = dKv @ Wkv + dScore @ Wgate`` and ``dW = dNew^T @ X``. APE partials are
reduced in a fixed core order, so the launch is deterministic.
"""

import tilelang.ascend.language as T

NUM_CORES = 32


def select_d_tile(window, cmp_ratio, coff, head_dim):
    """Largest head-dim tile whose softmax window fits in one UB."""
    budget = 200 * 1024
    tile = 256 if head_dim % 256 == 0 else 128
    while tile >= 32:
        # score, kv and dZ stay in fp32; dKv/dZ are also stored as the GEMM dtype.
        fp32_bytes = (3 * window + coff * cmp_ratio) * tile * 4
        bf_bytes = 2 * window * tile * 2
        if head_dim % tile == 0 and fp32_bytes + bf_bytes <= budget:
            return tile
        tile //= 2
    raise ValueError(f"compression window {window} does not fit in unified buffer")


def compressor_scatter(num_groups, tokens, head_dim, cmp_ratio, coff, dtype):
    """Map one compressed block's gradients onto token rows and an APE partial."""
    if coff not in (1, 2):
        raise ValueError(f"coff must be 1 or 2, got {coff}")
    window = coff * cmp_ratio
    compress_dim = coff * head_dim
    if tokens != num_groups * cmp_ratio:
        raise ValueError("token count must be num_groups * cmp_ratio")
    d_tile = select_d_tile(window, cmp_ratio, coff, head_dim)
    d_tiles = head_dim // d_tile
    threads = min(256, d_tile)
    steps = (num_groups + NUM_CORES - 1) // NUM_CORES

    @T.prim_func
    def main(
        d_cmp: T.Tensor((num_groups, head_dim), dtype),
        score: T.Tensor((num_groups, window, head_dim), "float32"),
        kv: T.Tensor((num_groups, window, head_dim), "float32"),
        meta: T.Tensor((num_groups, 5), "int32"),
        d_new_kv: T.Tensor((tokens, compress_dim), dtype),
        d_new_score: T.Tensor((tokens, compress_dim), dtype),
        partial: T.Tensor((NUM_CORES, cmp_ratio, compress_dim), "float32"),
    ):
        with T.Kernel(NUM_CORES) as bx:
            score_ub = T.alloc_shared((window, d_tile), "float32")
            kv_ub = T.alloc_shared((window, d_tile), "float32")
            dz_ub = T.alloc_shared((window, d_tile), "float32")
            dk_bf = T.alloc_shared((window, d_tile), dtype)
            dz_bf = T.alloc_shared((window, d_tile), dtype)
            dc_bf = T.alloc_shared((d_tile,), dtype)
            dc_fp = T.alloc_shared((d_tile,), "float32")
            ape_lo = T.alloc_shared((cmp_ratio, d_tile), "float32")
            zero_row = T.alloc_shared((compress_dim,), dtype)
            if coff == 2:
                ape_hi = T.alloc_shared((cmp_ratio, d_tile), "float32")
            T.clear(zero_row)

            for step in T.serial(steps):
                group = step * NUM_CORES + bx
                if group < num_groups:
                    cur = meta[group, 0]
                    valid = meta[group, 3]
                    if valid == 0:
                        for row in T.serial(cmp_ratio):
                            T.copy(zero_row, d_new_kv[cur + row, 0:compress_dim])
                            T.copy(zero_row, d_new_score[cur + row, 0:compress_dim])
                    elif coff == 2:
                        if meta[group, 4] == 1:
                            for row in T.serial(cmp_ratio):
                                T.copy(zero_row[0:head_dim], d_new_kv[cur + row, 0:head_dim])
                                T.copy(zero_row[0:head_dim], d_new_score[cur + row, 0:head_dim])

            for tile in T.serial(d_tiles):
                d0 = tile * d_tile
                T.clear(ape_lo)
                if coff == 2:
                    T.clear(ape_hi)
                for step in T.serial(steps):
                    group = step * NUM_CORES + bx
                    if group < num_groups:
                        if meta[group, 3] == 1:
                            cur = meta[group, 0]
                            prev = meta[group, 1]
                            ape_base = meta[group, 2]
                            T.copy(d_cmp[group, d0 : d0 + d_tile], dc_bf)
                            with T.SimtVF(threads=threads):
                                for col in T.Parallel(d_tile):
                                    dc_fp[col] = T.Cast("float32", dc_bf[col])
                            T.copy(score[group, 0:window, d0 : d0 + d_tile], score_ub)
                            T.copy(kv[group, 0:window, d0 : d0 + d_tile], kv_ub)
                            with T.SimtVF(threads=threads):
                                for col in T.Parallel(d_tile):
                                    acc = T.alloc_var("float32", init=T.float32(0))
                                    for row in T.serial(window):
                                        acc += score_ub[row, col] * dc_fp[col] * kv_ub[row, col]
                                    for row in T.serial(window):
                                        weighted = dc_fp[col] * kv_ub[row, col]
                                        dz_ub[row, col] = score_ub[row, col] * (weighted - acc)
                                        dk_bf[row, col] = T.Cast(dtype, dc_fp[col] * score_ub[row, col])
                                        dz_bf[row, col] = T.Cast(dtype, dz_ub[row, col])
                            if coff == 1:
                                for row in T.serial(cmp_ratio):
                                    ape_row = (ape_base + row) % cmp_ratio
                                    with T.SimtVF(threads=threads):
                                        for col in T.Parallel(d_tile):
                                            ape_lo[ape_row, col] = ape_lo[ape_row, col] + dz_ub[row, col]
                                    T.copy(dk_bf[row, 0:d_tile], d_new_kv[cur + row, d0 : d0 + d_tile])
                                    T.copy(dz_bf[row, 0:d_tile], d_new_score[cur + row, d0 : d0 + d_tile])
                            else:
                                for row in T.serial(cmp_ratio):
                                    ape_row = (ape_base + row) % cmp_ratio
                                    with T.SimtVF(threads=threads):
                                        for col in T.Parallel(d_tile):
                                            ape_lo[ape_row, col] = ape_lo[ape_row, col] + dz_ub[row * 2, col]
                                            ape_hi[ape_row, col] = ape_hi[ape_row, col] + dz_ub[row * 2 + 1, col]
                                    T.copy(
                                        dk_bf[row * 2 + 1, 0:d_tile],
                                        d_new_kv[cur + row, head_dim + d0 : head_dim + d0 + d_tile],
                                    )
                                    T.copy(
                                        dz_bf[row * 2 + 1, 0:d_tile],
                                        d_new_score[cur + row, head_dim + d0 : head_dim + d0 + d_tile],
                                    )
                                if prev >= 0:
                                    for row in T.serial(cmp_ratio):
                                        T.copy(dk_bf[row * 2, 0:d_tile], d_new_kv[prev + row, d0 : d0 + d_tile])
                                        T.copy(dz_bf[row * 2, 0:d_tile], d_new_score[prev + row, d0 : d0 + d_tile])
                T.copy(ape_lo, partial[bx, 0:cmp_ratio, d0 : d0 + d_tile])
                if coff == 2:
                    T.copy(ape_hi, partial[bx, 0:cmp_ratio, head_dim + d0 : head_dim + d0 + d_tile])

    return main


def compressor_reduce_ape(cmp_ratio, compress_dim):
    """Sum per-core APE partials in core order."""
    if compress_dim % NUM_CORES != 0:
        raise ValueError(f"compress dim {compress_dim} must be divisible by {NUM_CORES}")
    cols = compress_dim // NUM_CORES
    threads = min(256, max(32, cmp_ratio * cols))

    @T.prim_func
    def main(
        partial: T.Tensor((NUM_CORES, cmp_ratio, compress_dim), "float32"),
        d_ape: T.Tensor((cmp_ratio, compress_dim), "float32"),
    ):
        with T.Kernel(NUM_CORES) as bx:
            acc = T.alloc_shared((cmp_ratio, cols), "float32")
            buf = T.alloc_shared((cmp_ratio, cols), "float32")
            col0 = bx * cols
            T.clear(acc)
            for core in T.serial(NUM_CORES):
                T.copy(partial[core, 0:cmp_ratio, col0 : col0 + cols], buf)
                with T.SimtVF(threads=threads):
                    for row, col in T.Parallel(cmp_ratio, cols):
                        acc[row, col] = acc[row, col] + buf[row, col]
            T.copy(acc, d_ape[0:cmp_ratio, col0 : col0 + cols])

    return main


def _swizzle(m_tiles, n_tiles):
    window = min(4, m_tiles)
    main_row = m_tiles // window - 1
    tail_win = m_tiles - main_row * window

    @T.macro
    def aswt_swizzle(tile_idx):
        m_tile = T.alloc_var("int32")
        n_tile = T.alloc_var("int32")
        # The merged tail can span more than one regular window.
        row_idx = T.min(tile_idx // n_tiles // window, main_row)
        if row_idx < main_row:
            m_tile = row_idx * window + tile_idx % window
            n_tile = tile_idx // window % n_tiles
        else:
            tail_idx = tile_idx - main_row * window * n_tiles
            m_tile = main_row * window + tail_idx % tail_win
            n_tile = tail_idx // tail_win % n_tiles
        if row_idx % 2 != 0:
            n_tile = n_tiles - 1 - n_tile
        return (m_tile, n_tile)

    return aswt_swizzle


def compressor_dx(tokens, hidden, compress_dim, dtype, cube_blocks=NUM_CORES):
    """``dX = dKv @ Wkv + dScore @ Wgate`` with one fp32 accumulator."""
    tile_m = 256
    tile_n = 256
    tile_k = 256 if compress_dim % 256 == 0 else 128
    if tokens % tile_m or hidden % tile_n or compress_dim % tile_k:
        raise ValueError(f"dX tile ({tile_m}, {tile_n}, {tile_k}) does not divide {(tokens, hidden, compress_dim)}")
    m_tiles = tokens // tile_m
    n_tiles = hidden // tile_n
    k_tiles = compress_dim // tile_k
    out_tiles = m_tiles * n_tiles
    stages = 2 if k_tiles >= 2 else 0
    aswt_swizzle = _swizzle(m_tiles, n_tiles)
    uf_mid = 2
    uf_last = 3

    @T.prim_func
    def main(
        d_kv: T.Tensor((tokens, compress_dim), dtype),
        d_score: T.Tensor((tokens, compress_dim), dtype),
        wkv: T.Tensor((compress_dim, hidden), dtype),
        wgate: T.Tensor((compress_dim, hidden), dtype),
        d_x: T.Tensor((tokens, hidden), dtype),
    ):
        with T.Kernel(cube_blocks) as bx:
            a_l1 = T.alloc_l1((tile_m, tile_k), dtype)
            b_l1 = T.alloc_l1((tile_n, tile_k), dtype)
            res = T.alloc_l0c((tile_m, tile_n), "float32")
            for tile_idx in T.Persistent([out_tiles], cube_blocks, bx):
                m_tile, n_tile = aswt_swizzle(tile_idx)
                for kt in T.Pipelined(k_tiles, num_stages=stages):
                    T.copy(
                        d_kv[m_tile * tile_m : (m_tile + 1) * tile_m, kt * tile_k : (kt + 1) * tile_k],
                        a_l1,
                    )
                    # W is stored [K, N]. dn2nz makes the [N, K] operand GEMM expects.
                    T.copy(
                        wkv[kt * tile_k : (kt + 1) * tile_k, n_tile * tile_n : (n_tile + 1) * tile_n],
                        b_l1,
                        transpose=True,
                    )
                    # Wait for the previous tile's Fixpipe before reusing res.
                    T.gemm(a_l1, b_l1, res, transpose_B=True, clear_accum=kt == 0, unit_flag_ctrl=uf_mid)
                for kt in T.Pipelined(k_tiles, num_stages=stages):
                    T.copy(
                        d_score[m_tile * tile_m : (m_tile + 1) * tile_m, kt * tile_k : (kt + 1) * tile_k],
                        a_l1,
                    )
                    T.copy(
                        wgate[kt * tile_k : (kt + 1) * tile_k, n_tile * tile_n : (n_tile + 1) * tile_n],
                        b_l1,
                        transpose=True,
                    )
                    T.gemm(
                        a_l1,
                        b_l1,
                        res,
                        transpose_B=True,
                        clear_accum=False,
                        unit_flag_ctrl=T.Select(kt == k_tiles - 1, uf_last, uf_mid),
                    )
                T.copy(res, d_x[m_tile * tile_m, n_tile * tile_n], unit_flag_ctrl=uf_last)

    return main


def compressor_dw(tokens, hidden, compress_dim, dtype, cube_blocks=NUM_CORES):
    """Compute dWkv and dWgate while sharing each X tile in L1."""
    tile_m = 128
    tile_n = 256
    tile_k = 256
    if tokens % tile_k or hidden % tile_n or compress_dim % tile_m:
        raise ValueError(f"dW tile ({tile_m}, {tile_n}, {tile_k}) does not divide {(compress_dim, hidden, tokens)}")
    m_tiles = compress_dim // tile_m
    n_tiles = hidden // tile_n
    k_tiles = tokens // tile_k
    out_tiles = m_tiles * n_tiles
    stages = 2 if k_tiles >= 2 else 0
    aswt_swizzle = _swizzle(m_tiles, n_tiles)
    uf_mid = 2
    uf_last = 3

    @T.prim_func
    def main(
        d_kv: T.Tensor((tokens, compress_dim), dtype),
        d_score: T.Tensor((tokens, compress_dim), dtype),
        x: T.Tensor((tokens, hidden), dtype),
        d_wkv: T.Tensor((compress_dim, hidden), dtype),
        d_wgate: T.Tensor((compress_dim, hidden), dtype),
    ):
        with T.Kernel(cube_blocks) as bx:
            kv_l1 = T.alloc_l1((tile_m, tile_k), dtype)
            score_l1 = T.alloc_l1((tile_m, tile_k), dtype)
            x_l1 = T.alloc_l1((tile_n, tile_k), dtype)
            res_kv = T.alloc_l0c((tile_m, tile_n), "float32")
            res_score = T.alloc_l0c((tile_m, tile_n), "float32")
            for tile_idx in T.Persistent([out_tiles], cube_blocks, bx):
                m_tile, n_tile = aswt_swizzle(tile_idx)
                m0 = m_tile * tile_m
                n0 = n_tile * tile_n
                for kt in T.Pipelined(k_tiles, num_stages=stages):
                    k0 = kt * tile_k
                    T.copy(d_kv[k0 : k0 + tile_k, m0 : m0 + tile_m], kv_l1, transpose=True)
                    T.copy(d_score[k0 : k0 + tile_k, m0 : m0 + tile_m], score_l1, transpose=True)
                    T.copy(x[k0 : k0 + tile_k, n0 : n0 + tile_n], x_l1, transpose=True)
                    unit_flag = T.Select(kt == k_tiles - 1, uf_last, uf_mid)
                    T.gemm(
                        kv_l1,
                        x_l1,
                        res_kv,
                        transpose_B=True,
                        clear_accum=kt == 0,
                        unit_flag_ctrl=unit_flag,
                    )
                    T.gemm(
                        score_l1,
                        x_l1,
                        res_score,
                        transpose_B=True,
                        clear_accum=kt == 0,
                        unit_flag_ctrl=unit_flag,
                    )
                T.copy(res_kv, d_wkv[m0 : m0 + tile_m, n0 : n0 + tile_n], unit_flag_ctrl=uf_last)
                T.copy(res_score, d_wgate[m0 : m0 + tile_m, n0 : n0 + tile_n], unit_flag_ctrl=uf_last)

    return main
