#include "common.h"

int3_union get_nearby_point(int offset, int dim, int3_union point)
{
    int3_union nearby = point;
    nearby.elem[dim] += offset;
    return nearby;
}

int IDXloc(int3_union un, int dim, int3_union local_sizes) 
{
    un.elem[dim] += 4;
    return un.elem[0]*local_sizes.elem[1]*local_sizes.elem[2] + un.elem[1]*local_sizes.elem[2] + un.elem[2];
}
 
 
kernel void
compute_second_function(
    global const T* zeta,
    const T delta,
    const int dimension,
    global T* result,
    local T* local_grid,
    const int nt,
    const int nx,
    const int ny
) {
    #define IDX(un) (un.elem[0]*nx*ny + un.elem[1]*ny + un.elem[2])

    const int nt_real = get_global_size(0);
    const int nx_real = get_global_size(1);
    const int ny_real = get_global_size(2);
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);

    if (i >= nt) return;
    if (j >= nx) return;
    if (k >= ny) return;
    
    const int loci = get_local_id(0);
    const int locj = get_local_id(1);
    const int lock = get_local_id(2);
    int loct = get_local_size(0);
    int locx = get_local_size(1);
    int locy = get_local_size(2);

    if (get_group_id(0) == get_num_groups(0) - 1 && nt % loct != 0)
        loct = nt % loct;
    if (get_group_id(1) == get_num_groups(1) - 1 && nx % locx != 0)
        locx = nx % locx;
    if (get_group_id(2) == get_num_groups(2) - 1 && ny % locy != 0)
        locy = ny % locy;
    
    int3_union local_sizes = { .vec = (int3)(loct, locx, locy) };
    int3_union       sizes = { .vec = (int3)(nt, nx, ny)};
    const int local_x = get_local_id(dimension);
    const int global_x = get_global_id(dimension);
    const int size = sizes.elem[dimension];
    const int block_x = local_sizes.elem[dimension];
    local_sizes.elem[dimension] += 8; //halo
    loct = local_sizes.elem[0];
    locx = local_sizes.elem[1];
    locy = local_sizes.elem[2];



    const int3_union idx0 = { .vec = (int3)(i, j, k) };
    const int global_id = IDX(idx0);
    int3_union idx_loc_vec = { .vec = (int3)(loci, locj, lock) };
    const T denominator = delta;
    for (int ii = local_x; ii < block_x +4 ; ii += block_x)
    {
        int3_union point = get_nearby_point(ii - local_x, dimension, idx_loc_vec);
        int3_union global_point = get_nearby_point(ii - local_x, dimension, idx0);
        local_grid[IDXloc(point, dimension, local_sizes)] = zeta[IDX(global_point)];
    }
 
    for (int ii = local_x - block_x; ii > -5; ii -= block_x)
    {
        int3_union point = get_nearby_point(ii - local_x, dimension, idx_loc_vec);
        int3_union global_point = get_nearby_point(ii - local_x, dimension, idx0);
        local_grid[IDXloc(point, dimension, local_sizes)] = zeta[IDX(global_point)];
    }
 
 
    barrier(CLK_LOCAL_MEM_FENCE);
 
    if (global_x > 3 && global_x < size - 4) {
        const T ax = 4.f / 5.f / denominator;
        const T bx = - 1.f / 5.f / denominator;
        const T cx = 4.f / 105.f / denominator;
        const T dx = -1.f / 280.f / denominator;
 
        int3_union left_dx_1 = get_nearby_point(-1, dimension, idx_loc_vec);
        int3_union left_dx_2 = get_nearby_point(-2, dimension, idx_loc_vec);
        int3_union left_dx_3 = get_nearby_point(-3, dimension, idx_loc_vec);
        int3_union left_dx_4 = get_nearby_point(-4, dimension, idx_loc_vec);
        int3_union right_dx_1 = get_nearby_point(1, dimension, idx_loc_vec);
        int3_union right_dx_2 = get_nearby_point(2, dimension, idx_loc_vec);
        int3_union right_dx_3 = get_nearby_point(3, dimension, idx_loc_vec);
        int3_union right_dx_4 = get_nearby_point(4, dimension, idx_loc_vec);
 
        result[global_id] = (
            ax * (local_grid[IDXloc(right_dx_1, dimension, local_sizes)] - local_grid[IDXloc(left_dx_1, dimension, local_sizes)]) +
            bx * (local_grid[IDXloc(right_dx_2, dimension, local_sizes)] - local_grid[IDXloc(left_dx_2, dimension, local_sizes)]) +
            cx * (local_grid[IDXloc(right_dx_3, dimension, local_sizes)] - local_grid[IDXloc(left_dx_3, dimension, local_sizes)]) +
            dx * (local_grid[IDXloc(right_dx_4, dimension, local_sizes)] - local_grid[IDXloc(left_dx_4, dimension, local_sizes)])
                );
    } else if (global_x <= 3) {
        const T ax = -25.f / 12.f / denominator;
        const T bx = 4.f  / denominator;
        const T cx = -3.f / denominator;
        const T dx = 4.f / 3.f / denominator;
        const T ex = -1.f / 4.f / denominator;
        int3_union left_dx_1 = get_nearby_point(1, dimension, idx_loc_vec);
        int3_union left_dx_2 = get_nearby_point(2, dimension, idx_loc_vec);
        int3_union left_dx_3 = get_nearby_point(3, dimension, idx_loc_vec);
        int3_union left_dx_4 = get_nearby_point(4, dimension, idx_loc_vec);
        result[global_id] =  ax * local_grid[IDXloc(idx_loc_vec, dimension, local_sizes)] +
                              bx * local_grid[IDXloc(left_dx_1, dimension, local_sizes)] +
                              cx * local_grid[IDXloc(left_dx_2, dimension, local_sizes)] +
                              dx * local_grid[IDXloc(left_dx_3, dimension, local_sizes)] +
                              ex * local_grid[IDXloc(left_dx_4, dimension, local_sizes)];
 
    } else if (global_x >= size - 4)
    {
        const T ax = -25.f / 12.f / denominator;
        const T bx = 4.f  / denominator;
        const T cx = -3.f / denominator;
        const T dx = 4.f / 3.f / denominator;
        const T ex = -1.f / 4.f / denominator;
        int3_union right_dx_1 = get_nearby_point(-1, dimension, idx_loc_vec);
        int3_union right_dx_2 = get_nearby_point(-2, dimension, idx_loc_vec);
        int3_union right_dx_3 = get_nearby_point(-3, dimension, idx_loc_vec);
        int3_union right_dx_4 = get_nearby_point(-4, dimension, idx_loc_vec);
        result[global_id] = -ax * local_grid[IDXloc(idx_loc_vec, dimension, local_sizes)] -
                              bx * local_grid[IDXloc(right_dx_1, dimension, local_sizes)] -
                              cx * local_grid[IDXloc(right_dx_2, dimension, local_sizes)] -
                              dx * local_grid[IDXloc(right_dx_3, dimension, local_sizes)] -
                              ex * local_grid[IDXloc(right_dx_4, dimension, local_sizes)];
 
    }
    #undef IDX
}
 
