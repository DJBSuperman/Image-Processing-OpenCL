Histogram Equalization with OpenCL

Divine Jacob - Parallel Programming 

Overview: 
This project implements histogram equalization for both 8-bit and 16-bit grayscale images, using OpenCL to accelerate the processing.

Upon execution, the application:

Automatically detects the bit depth of a loaded grayscale image.

Computes both the standard histogram and cumulative histogram using prefix sum (scan).

Normalizes the histogram to produce a Lookup Table (LUT).

Applies the LUT to enhance image contrast.

Displays the original and processed images alongside their histograms.

This parallelized approach ensures performance benefits on both CPU and GPU platforms.

📁 Features:


✅ Bit Depth Detection
The system dynamically determines whether the input is 8-bit or 16-bit by checking the maximum pixel value. This avoids manual configuration and supports seamless processing for either image type.


⚙️ OpenCL Kernels (Kernels.cl)
Separate kernel functions are implemented for 8-bit and 16-bit histograms.


The atomic_inc operation ensures race conditions are avoided during histogram bin updates.


📊 Histogram Computation
Parallelized histogram computation assigns each pixel to a work item. This boosts performance while maintaining accuracy via atomic operations.


➕ Prefix Sum (Scan)
A cumulative histogram is computed via a scan kernel to generate the Cumulative Distribution Function (CDF).


🧮 Histogram Normalization
CDF results are normalized and mapped to the output range, forming the LUT used to adjust pixel intensities.


📈 Visualization
The following are displayed:

Original grayscale image

Equalized image

Standard histogram

Cumulative histogram

Normalized histogram

🧠 Optimizations:


Avoids unnecessary work for 8-bit images by tailoring kernel logic.


Efficient use of CL_MEM_READ_ONLY for OpenCL buffer flags improves memory handling.


Kernel separation and parameter tuning enhance overall performance.


🧮 Memory Management
Device buffers are allocated for:


Input image


Histogram data


Prefix scan results


Normalized LUT


These optimizations ensure better runtime memory behavior.


⏱️ Performance Metrics
Here are sample performance timings showing execution under different conditions:


Test	Total Execution	Memory Transfer	Histogram Kernel	Scan Kernel	Equalize Kernel
Run 1	698.996 ms	0.976 ms	0.276 ms	1.839 ms	0.000 ms
Run 2	521.163 ms	0.242 ms	0.196 ms	0.024 ms	0.000 ms
Run 3	1075.56 ms	2.691 ms	4.780 ms	0.025 ms	0.000 ms


These timings demonstrate the impact of parallelization, memory optimization, and adaptive bit depth handling.


🚀 Key Parallelization Strategies:


Global Work Item Distribution: Each work item handles one pixel.

Separation of Kernels: Optimized individual kernels for histogram, scan, and equalization.

Efficient Equalization Step: The LUT application is minimal in computational cost but crucial for image contrast.

This design focuses computation where it delivers the most benefit.


🛠️ Technologies Used
OpenCL for parallel kernel execution


C/C++ as the host language


Compatible with CPU/GPU devices


📌 Final Thoughts:


This project highlights the effectiveness of OpenCL in real-world image processing. Bit-depth-aware kernels, memory optimizations, and parallel strategies combined to improve histogram equalization runtime without compromising on visual output quality.

