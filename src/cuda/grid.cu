#include <iostream>
#include <cuda_runtime.h>

extern "C" {
    
    // fills a 3D grid with a specified value
    void fill_grid(unsigned int* grid, int width, int height, int slots, float value) {
        int size = width * height * slots;
        /// TODOOO
    }

}