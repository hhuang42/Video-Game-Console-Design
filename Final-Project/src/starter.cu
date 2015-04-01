// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
// http://computer-graphics.se/hello-world-for-cuda.html
 
#include <stdio.h>
 
const int N = 16; 
const int block_width = 4;
const int block_height = 4;
 
 
typedef struct bullet_t{
    double x;
    double y;
    double x_v;
    double y_v;
} basic_bullet;
 

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

__device__ void initialize_bullet(basic_bullet* bullet,
    double x, double y, double x_v, double y_v){
    bullet->x = x;
    bullet->y = y;
    bullet->x_v = x_v;
    bullet->y_v = y_v;
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

__device__ void update_bullet(basic_bullet* bullet){
    bullet->x += bullet->x_v;
    bullet->y += bullet->y_v;
}
 
int main()
{
	char a[N] = "Hello \0\0\0\0\0\0";
	int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 
 char *ad;
 int *bd;
 const int csize = N*sizeof(char);
 const int isize = N*sizeof(int);
 
 printf("%s", a);
 
 cudaMalloc( (void**)&ad, csize ); 
 cudaMalloc( (void**)&bd, isize ); 
 cudaMemcpyAsync( ad, a, csize, cudaMemcpyHostToDevice ); 
 cudaMemcpyAsync( bd, b, isize, cudaMemcpyHostToDevice ); 
 
 dim3 dimBlock( blocksize, 1 );
 dim3 dimGrid( 1, 1 );
 hello<<<dimGrid, dimBlock>>>(ad, bd);
 cudaMemcpyAsync( a, ad, csize, cudaMemcpyDeviceToHost ); 
 cudaFree( ad );
 cudaFree( bd );
 
 printf("%s\n", a);
 return EXIT_SUCCESS;
}


