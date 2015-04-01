// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
// http://computer-graphics.se/hello-world-for-cuda.html
 
#include <stdio.h>
 
const int block_size = 16; 
const int block_width = 16;
const int block_height = 1;
 
 
typedef struct bullet_t{
    double x;
    double y;
    double x_v;
    double y_v;
} basic_bullet;

typedef struct position_t{
    double x;
    double y;
} position;
 
__device__ void initialize_bullet(basic_bullet* bullet,
    double x, double y, double x_v, double y_v){
    bullet->x = x;
    bullet->y = y;
    bullet->x_v = x_v;
    bullet->y_v = y_v;
}

__global__ void initialize_bullets(basic_bullet* bullets,
                                   size_t bullet_count,
                                   size_t total_threads){
    int bullet_index = threadIdx.x;
    while(bullet_index < bullet_count){
        initialize_bullet(bullets + bullet_index,
        threadIdx.x, threadIdx.y,
        .01, -.01);
        bullet_index += total_threads;
    }
}

__device__ void update_bullet(basic_bullet* bullet){
    bullet->x += bullet->x_v;
    bullet->y += bullet->y_v;
}

__global__ void update_bullets(basic_bullet* bullets,
                               size_t bullet_count,
                               size_t total_threads)
{
    int bullet_index = threadIdx.x;
    while(bullet_index < bullet_count){
        update_bullet(bullets + bullet_index);
        bullet_index += total_threads;
    }
}

__device__ void transfer_bullet_position(basic_bullet* bullet,
                                         position* output){
    output->x = bullet->x;
    output->y = bullet->y;
}

__global__ void transfer_bullets_position(basic_bullet* bullets,
                                          position* output,
                                          size_t bullet_count,
                                          size_t total_threads){
    int bullet_index = threadIdx.x;
    while(bullet_index < bullet_count){
        transfer_bullet_position(bullets + bullet_index
                                 output + bullet_index);
        bullet_index += total_threads;
    }    
}


 
int main()
{
 
    basic_bullet* bullets_d;
    position* positions_d;
    position* positions_h;
    
     
    const int bullets_count = 1000;
    const int bullets_size = bullets_count*sizeof(basic_bullet);
    const int positions_size = bullets_count*sizeof(position);
    
    dim3 dimBlock( block_width, block_height );
    dim3 dimGrid( 1, 1 );
 
// printf("%s", a);
// 
    cudaMalloc( (void**)&bullets_d, bullets_size );
    cudaMalloc( (void**)&bullets_d, bullets_size );
    positions_h malloc()
    initialize_bullets<<<dimGrid, dimBlock>>>(
                                   bullets_d,
                                   bullets_count,
                                   block_size);
    initialize_bullets<<<dimGrid, dimBlock>>>(
                                   bullets_d,
                                   bullets_count,
                                   block_size);
    cudaFree(bullets_d);
    
// cudaMalloc( (void**)&bd, isize ); 
// cudaMemcpyAsync( ad, a, csize, cudaMemcpyHostToDevice ); 
// cudaMemcpyAsync( bd, b, isize, cudaMemcpyHostToDevice ); 
// 
// dim3 dimBlock( blocksize, 1 );
// dim3 dimGrid( 1, 1 );
// hello<<<dimGrid, dimBlock>>>(ad, bd);
// cudaMemcpyAsync( a, ad, csize, cudaMemcpyDeviceToHost ); 
// cudaFree( ad );
// cudaFree( bd );
// 
// printf("%s\n", a);
// return EXIT_SUCCESS;
}


