#include "common.h"

int3_union get_nearby_point(int offset, int dim, int3_union point)
{
	int3_union nearby = point;
	nearby.elem[dim] += offset;
	return nearby;	
}


kernel void
compute_second_function(
	global const T* zeta,
	global const T* delta,
	const int dimension,
	global T* result,
	local T* local_grid
) {
	#define IDX(un) (un.elem[0]*nx*ny + un.elem[1]*ny + un.elem[2])
	#define IDXloc(un) (un.elem[0]*locx*locy + un.elem[1]*locy + un.elem[2])
	const int nt = get_global_size(0);
	const int nx = get_global_size(1);
	const int ny = get_global_size(2);
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	const int k = get_global_id(2);

	const int loci = get_local_id(0);
	const int locj = get_local_id(1);
	const int lock = get_local_id(2);
	int loct = get_local_size(0);
	int locx = get_local_size(1);
	int locy = get_local_size(2);

	const int local_x = get_local_id(dimension);
	const int block_x = get_local_size(dimension);
	const int global_x = get_global_id(dimension);
	const int size   = get_global_size(dimension);
	int3_union dim_sizes = { .vec = (int3)(loct, locx, locy) };
	dim_sizes.elem[dimension] += 8; //halo
	loct = dim_sizes.elem[0];
	locx = dim_sizes.elem[1];	
	locy = dim_sizes.elem[2];	 
	const int3_union idx0 = { .vec = (int3)(i, j, k) };
	const int global_id = IDX(idx0);
	int3_union idx_loc_vec = { .vec = (int3)(loci, locj, lock) };
	const T denominator = /* 2* */delta[dimension];
	for (int i = local_x; i < block_x + 4; i += block_x)
	{
		if (i - local_x < size)
		{
			int3_union point = get_nearby_point(i - local_x, dimension, idx_loc_vec);
			int3_union global_point = get_nearby_point(i - local_x, dimension, idx0);
			local_grid[IDXloc(point)] = zeta[IDX(global_point)]; 
		}
	}
	
	for (int i = local_x - block_x; i > -5; i -= block_x)
	{
		if (i - local_x >= 0)
		{
			int3_union point = get_nearby_point(i - local_x, dimension, idx_loc_vec);
			int3_union global_point = get_nearby_point(i - local_x, dimension, idx0);
			local_grid[IDXloc(point)] = zeta[IDX(global_point)]; 
		}
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
			ax * (local_grid[IDXloc(right_dx_1)] - local_grid[IDXloc(left_dx_1)]) +
			bx * (local_grid[IDXloc(right_dx_2)] - local_grid[IDXloc(left_dx_2)]) +
			cx * (local_grid[IDXloc(right_dx_3)] - local_grid[IDXloc(left_dx_3)]) +
			dx * (local_grid[IDXloc(right_dx_4)] - local_grid[IDXloc(left_dx_4)])  
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
		result[global_id] =  ax * local_grid[IDXloc(idx_loc_vec)] + 
							  bx * local_grid[IDXloc(left_dx_1)] + 
							  cx * local_grid[IDXloc(left_dx_2)] + 
							  dx * local_grid[IDXloc(left_dx_3)] + 
							  ex * local_grid[IDXloc(left_dx_4)];
		
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
		result[global_id] = -ax * local_grid[IDXloc(idx_loc_vec)] - 
							  bx * local_grid[IDXloc(right_dx_1)] - 
							  cx * local_grid[IDXloc(right_dx_2)] - 
							  dx * local_grid[IDXloc(right_dx_3)] - 
							  ex * local_grid[IDXloc(right_dx_4)];
		
	}
	#undef IDX
    #undef IDXloc
}

