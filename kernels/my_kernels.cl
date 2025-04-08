// Histogram calculation kernel for grayscale images
kernel void histogram_map(global const uchar* A, global int* histogram, const int width, const int height) {
    int id = get_global_id(0);
    
    // Ensure the thread is within bounds
    if (id < width * height) {
        // Extract pixel intensity (already grayscale)
        uchar pixel_intensity = A[id]; 
        // Increment corresponding histogram bin
        atomic_inc(&histogram[pixel_intensity]);
    }
}

// Histogram reduction kernel
kernel void histogram_reduce(global int* histogram, global int* reduced_histogram, const int histogram_size) {
    int id = get_global_id(0);
    // Make sure the thread index is within bounds
    if (id < histogram_size) {
        reduced_histogram[id] = histogram[id];
    }
}
