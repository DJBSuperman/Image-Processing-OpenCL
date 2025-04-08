#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;
using namespace std;

void print_help() {
    std::cerr << "Application usage:" << std::endl;

    std::cerr << "  -p : select platform " << std::endl;
    std::cerr << "  -d : select device" << std::endl;
    std::cerr << "  -l : list all platforms and devices" << std::endl;
    std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
    std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
    // Part 1 - handle command line options such as device selection, verbosity, etc.
    int platform_id = 0;
    int device_id = 0;
    string image_filename = "test.pgm";

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
        else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
        else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
        else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
    }

    cimg::exception_mode(0);

    // Detect any potential exceptions
    try {
        CImg<unsigned char> image_input(image_filename.c_str());
        CImgDisplay disp_input(image_input, "Input Image");

        // Part 3 - host operations
        // 3.1 Select computing devices
        cl::Context context = GetContext(platform_id, device_id);

        // Display the selected device
        std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

        // Create a queue to which we will push commands for the device
        cl::CommandQueue queue(context);

        // 3.2 Load & build the device code
        cl::Program::Sources sources;
        AddSources(sources, "kernel");  // Update to correct kernel file name

        cl::Program program(context, sources);

        // Build and debug the kernel code
        try {
            program.build();
        } catch (const cl::Error& err) {
            std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            throw err;
        }

        // Part 4 - device operations
        // Device - buffers
        cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());

        // 4.1 Copy images to device memory
        queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);

        // Create histogram images with additional space for axes (300x300 instead of 256x256)
        CImg<unsigned char> histogram_img(300, 300, 1, 3, 255);  // RGB image for better visualization
        CImg<unsigned char> cumulative_img(300, 300, 1, 3, 255);  // RGB image for better visualization

        // Compute histogram using OpenCL kernel
        const int histogram_size = 256;
        vector<int> host_histogram(histogram_size, 0);
        
        // Create device buffers for histogram computation
        cl::Buffer dev_histogram(context, CL_MEM_READ_WRITE, histogram_size * sizeof(int));
        
        // Initialize histogram to zeros
        queue.enqueueFillBuffer(dev_histogram, 0, 0, histogram_size * sizeof(int));
        
        // Setup and execute the histogram_map kernel
        cl::Kernel hist_kernel = cl::Kernel(program, "histogram_map");
        hist_kernel.setArg(0, dev_image_input);  // Use the grayscale image directly
        hist_kernel.setArg(1, dev_histogram);
        hist_kernel.setArg(2, image_input.width());
        hist_kernel.setArg(3, image_input.height());
        
        queue.enqueueNDRangeKernel(hist_kernel, cl::NullRange, cl::NDRange(image_input.width() * image_input.height()), cl::NullRange);
        
        // Read back the histogram results
        queue.enqueueReadBuffer(dev_histogram, CL_TRUE, 0, histogram_size * sizeof(int), host_histogram.data());

        // Create the cumulative histogram
        std::vector<int> cumulative_histogram(256, 0);
        cumulative_histogram[0] = host_histogram[0];
        for (int i = 1; i < 256; i++) {
            cumulative_histogram[i] = cumulative_histogram[i - 1] + host_histogram[i];
        }

        // Find the maximum value in the histogram for normalization
        int max_hist_val = 0;
        for (int i = 0; i < 256; i++) {
            if (host_histogram[i] > max_hist_val) {
                max_hist_val = host_histogram[i];
            }
        }
        
        // Find the maximum value in the cumulative histogram
        int max_cum_val = cumulative_histogram[255];

        std::cout << "Max histogram value: " << max_hist_val << std::endl;
        std::cout << "Max cumulative histogram value: " << max_cum_val << std::endl;

        // Cap max_hist_val to 2000 for Y-axis scaling
        max_hist_val = std::min(max_hist_val, 2000);

        // Draw axes on histogram image
        const unsigned char black[] = {0, 0, 0};
        const unsigned char gray[] = {200, 200, 200};
        const unsigned char red[] = {255, 0, 0};
        
        // X-axis
        histogram_img.draw_line(30, 270, 270, 270, black);
        // Y-axis
        histogram_img.draw_line(30, 30, 30, 270, black);
        
        // Draw grid lines on histogram
        for (int i = 1; i <= 10; i++) {
            int y = 270 - i * 24;  // 24 pixels per 200 value increment (240/10)
            histogram_img.draw_line(28, y, 270, y, gray);
            
            // Y-axis labels (0, 200, 400, ..., 2000)
            char label[10];
            sprintf(label, "%d", i * 200);
            histogram_img.draw_text(5, y - 5, label, black);
        }
        
        // X-axis labels (0, 50, 100, ..., 250)
        for (int i = 0; i <= 5; i++) {
            int x = 30 + i * 48;  // 48 pixels per 50 value increment (240/5)
            histogram_img.draw_line(x, 270, x, 272, black);
            
            char label[10];
            sprintf(label, "%d", i * 50);
            histogram_img.draw_text(x - 10, 280, label, black);
        }
        
        // Draw histogram bars in red
        for (int i = 0; i < 256; i++) {
            // Scale histogram values to fit in 0-2000 range
            int hist_val = host_histogram[i];
            hist_val = std::min(hist_val, 2000);  // Cap at 2000 for display
            
            // Convert to pixel coordinates (30-270 x range, 30-270 y range)
            int x = 30 + (i * 240) / 255;
            int bar_height = (hist_val * 240) / 2000;
            
            if (bar_height > 0) {
                histogram_img.draw_line(x, 270, x, 270 - bar_height, red);
            }
        }
        
        // Draw axes on cumulative histogram
        // X-axis
        cumulative_img.draw_line(30, 270, 270, 270, black);
        // Y-axis
        cumulative_img.draw_line(30, 30, 30, 270, black);
        
        // Scale cumulative histogram to have a reasonable display
        int cum_scale = (max_cum_val > 0) ? max_cum_val : 1;
        
        // Draw grid lines on cumulative histogram
        for (int i = 1; i <= 10; i++) {
            int y = 270 - i * 24;  // 24 pixels per division
            cumulative_img.draw_line(28, y, 270, y, gray);
            
            // Y-axis labels (10%, 20%, ..., 100% of max)
            char label[10];
            sprintf(label, "%d%%", i * 10);
            cumulative_img.draw_text(5, y - 5, label, black);
        }
        
        // X-axis labels (0, 50, 100, ..., 250)
        for (int i = 0; i <= 5; i++) {
            int x = 30 + i * 48;  // 48 pixels per 50 value increment
            cumulative_img.draw_line(x, 270, x, 272, black);
            
            char label[10];
            sprintf(label, "%d", i * 50);
            cumulative_img.draw_text(x - 10, 280, label, black);
        }
        
        // Draw cumulative histogram as a blue line
        const unsigned char blue[] = {0, 0, 255};
        int prev_x = 30;
        int prev_y = 270;
        
        for (int i = 0; i < 256; i++) {
            // Convert to pixel coordinates (30-270 x range, 30-270 y range)
            int x = 30 + (i * 240) / 255;
            int y = 270 - (cumulative_histogram[i] * 240) / cum_scale;
            
            // Draw line segment to connect points
            cumulative_img.draw_line(prev_x, prev_y, x, y, blue);
            prev_x = x;
            prev_y = y;
        }

        // Display the histograms
        CImgDisplay disp_histogram(histogram_img, "Histogram (0-2000)");
        CImgDisplay disp_cumulative(cumulative_img, "Cumulative Histogram (%)");

        // Wait for all windows to be closed
        while (!disp_input.is_closed() && !disp_histogram.is_closed() && !disp_cumulative.is_closed()) {
            disp_input.wait(1);
            disp_histogram.wait(1);
            disp_cumulative.wait(1);
        }
    } catch (const cl::Error& err) {
        std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
    } catch (CImgException& err) {
        std::cerr << "ERROR: " << err.what() << std::endl;
    }

    return 0;
}
