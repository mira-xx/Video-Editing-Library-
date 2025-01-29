// Copyright 2024 <Mohana Maran>

#include "FilmMaster2000.h"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <immintrin.h>
#include <utility>

Video FilmMaster2000::decode(const std::string& filename) {
    Video video;

    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary | std::ios::in);
    if (!file) {
        throw std::runtime_error("Failed to open input file");
    }

    // Read metadata
    file.read(reinterpret_cast<char*>(&video.numFrames), sizeof(video.numFrames));
    file.read(reinterpret_cast<char*>(&video.channels), sizeof(video.channels));
    file.read(reinterpret_cast<char*>(&video.height), sizeof(video.height));
    file.read(reinterpret_cast<char*>(&video.width), sizeof(video.width));

    size_t frameSize = video.channels * video.height * video.width;
    size_t totalSize = frameSize * video.numFrames;
    video.frames.resize(video.numFrames);

    // Read all file data into a single buffer
    std::vector<uint8_t> buffer(totalSize);
    file.read(reinterpret_cast<char*>(buffer.data()), totalSize);
    file.close();

    // Copy data from the buffer to frames
    #pragma omp parallel for
    for (size_t i = 0; i < static_cast<size_t>(video.numFrames); ++i) {
        video.frames[i].data = std::make_unique<uint8_t[]>(frameSize);   // Allocate memory for the frame
        std::memcpy(video.frames[i].data.get(), buffer.data() + i * frameSize, frameSize);   // Copy data from the buffer
    }

    return video;
}

Video FilmMaster2000::decode_speed(const std::string& filename) {
    Video video;

    std::ifstream file(filename, std::ios::binary | std::ios::in);
    if (!file) {
        throw std::runtime_error("Failed to open input file");
    }

    file.read(reinterpret_cast<char*>(&video.numFrames), sizeof(video.numFrames));
    file.read(reinterpret_cast<char*>(&video.channels), sizeof(video.channels));
    file.read(reinterpret_cast<char*>(&video.height), sizeof(video.height));
    file.read(reinterpret_cast<char*>(&video.width), sizeof(video.width));

    size_t frameSize = video.channels * video.height * video.width;
    video.frames.resize(video.numFrames);


    auto decode_task = [&](size_t start, size_t end) {
        // Open file for reading in each thread
        std::ifstream threadFile(filename, std::ios::binary | std::ios::in);
        // Move to the start of the frame data
        threadFile.seekg(sizeof(video.numFrames) + sizeof(video.channels) + sizeof(video.height) + sizeof(video.width), std::ios::beg);

        // Calculate size of the metedata header and the offset
        // to allow the thread to start reading from the correct position
        size_t headerSize = sizeof(video.numFrames) + sizeof(video.channels) + sizeof(video.height) + sizeof(video.width);
        size_t fileOffset = headerSize + start * frameSize;

        threadFile.seekg(fileOffset, std::ios::beg);   // Move to the calculated starting position

        for (size_t i = start; i < end; ++i) {
            auto& frame = video.frames[i];
            frame.data = std::make_unique<uint8_t[]>(frameSize);
            threadFile.read(reinterpret_cast<char*>(frame.data.get()), frameSize);

            __m128i* simdPtr = reinterpret_cast<__m128i*>(frame.data.get());   // Set up the pointer for SIMD operations
            size_t simdCount = frameSize / 16;

            for (size_t j = 0; j < simdCount; ++j) {
                simdPtr[j] = _mm_loadu_si128(simdPtr + j);
            }
        }
    };

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 2;

    size_t framesPerThread = video.numFrames / numThreads;   // Number of frames per thread
    size_t remainingFrames = video.numFrames % numThreads;

    // Launch threads to decode frames
    std::vector<std::thread> threads;
    size_t start = 0;
    for (unsigned int t = 0; t < numThreads; ++t) {
        size_t end = start + framesPerThread + (t < remainingFrames ? 1 : 0);
        threads.emplace_back(decode_task, start, end);
        start = end;
    }

    for (auto& thread : threads) {
        thread.join();
    }

    file.close();
    return video;
}

Video FilmMaster2000::decode_memory(const std::string& filename) {
    Video video;

    // Open the file
    std::ifstream file(filename, std::ios::binary | std::ios::in);
    if (!file) {
        throw std::runtime_error("Failed to open input file");
    }

    // Read metadata
    file.read(reinterpret_cast<char*>(&video.numFrames), sizeof(video.numFrames));
    file.read(reinterpret_cast<char*>(&video.channels), sizeof(video.channels));
    file.read(reinterpret_cast<char*>(&video.height), sizeof(video.height));
    file.read(reinterpret_cast<char*>(&video.width), sizeof(video.width));

    size_t frameSize = video.channels * video.height * video.width;

    // Reserve space for frames metadata
    video.frames.reserve(video.numFrames);

    // Process frames one at a time
    for (size_t i = 0; i < static_cast<size_t>(video.numFrames); ++i) {
        // Allocate memory for the current frame
        auto frameData = std::make_unique<uint8_t[]>(frameSize);

        // Read the frame data directly into the frame buffer
        file.read(reinterpret_cast<char*>(frameData.get()), frameSize);
        if (!file) {
            throw std::runtime_error("Failed to read frame data");
        }

        // Store the frame data in the video object
        Frame frame;
        frame.data = std::move(frameData);
        video.frames.push_back(std::move(frame));
    }

    file.close();
    return video;
}

void FilmMaster2000::encode(const std::string& filename, const Video& video) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open output file");
    }

    // Write video metadata
    file.write(reinterpret_cast<const char*>(&video.numFrames), sizeof(video.numFrames));
    file.write(reinterpret_cast<const char*>(&video.channels), sizeof(video.channels));
    file.write(reinterpret_cast<const char*>(&video.height), sizeof(video.height));
    file.write(reinterpret_cast<const char*>(&video.width), sizeof(video.width));

    // Write frames
    size_t frameSize = video.channels * video.height * video.width;
    for (const auto& frame : video.frames) {
        file.write(reinterpret_cast<const char*>(frame.data.get()), frameSize);
    }

    file.close();
}

void FilmMaster2000::encode_memory(const std::string& filename, Video& video) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open output file");
    }

    // Write video metadata
    file.write(reinterpret_cast<const char*>(&video.numFrames), sizeof(video.numFrames));
    file.write(reinterpret_cast<const char*>(&video.channels), sizeof(video.channels));
    file.write(reinterpret_cast<const char*>(&video.height), sizeof(video.height));
    file.write(reinterpret_cast<const char*>(&video.width), sizeof(video.width));

    // Optimize writes with a single buffer
    size_t frameSize = video.channels * video.height * video.width;
    std::unique_ptr<char[]> buffer(new char[frameSize]);

    for (auto& frame : video.frames) {
        std::memcpy(buffer.get(), frame.data.get(), frameSize);
        file.write(buffer.get(), frameSize);
    }

    // Clear frames after all writes
    video.frames.clear();
    video.frames.shrink_to_fit();

    file.close();
}

void FilmMaster2000::encode_speed(const std::string& filename, const Video& video) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open output file");
    }

    // Write video metadata
    file.write(reinterpret_cast<const char*>(&video.numFrames), sizeof(video.numFrames));
    file.write(reinterpret_cast<const char*>(&video.channels), sizeof(video.channels));
    file.write(reinterpret_cast<const char*>(&video.height), sizeof(video.height));
    file.write(reinterpret_cast<const char*>(&video.width), sizeof(video.width));

    size_t frameSize = video.channels * video.height * video.width;

    // Allocate separate buffers for each thread
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 2;

    size_t framesPerThread = video.numFrames / numThreads;
    size_t remainingFrames = video.numFrames % numThreads;

    std::vector<std::thread> threads;
    std::vector<std::unique_ptr<char[]>> threadBuffers(numThreads);

    // Lambda to encode a chunk of frames
    auto encode_function = [&](size_t start, size_t end, char* buffer) {
        size_t bufferOffset = 0;
        for (size_t i = start; i < end; ++i) {
            std::memcpy(buffer + bufferOffset, video.frames[i].data.get(), frameSize);
            bufferOffset += frameSize;
        }
    };

    // Launch threads to populate buffers
    size_t start = 0;
    for (unsigned int t = 0; t < numThreads; ++t) {
        size_t end = start + framesPerThread + (t < remainingFrames ? 1 : 0);
        threadBuffers[t] = std::make_unique<char[]>((end - start) * frameSize);
        threads.emplace_back(encode_function, start, end, threadBuffers[t].get());
        start = end;
    }

    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Write all buffers sequentially
    for (unsigned int t = 0; t < numThreads; ++t) {
        size_t bufferSize = ((t < remainingFrames ? framesPerThread + 1 : framesPerThread)) * frameSize;
        file.write(threadBuffers[t].get(), bufferSize);
    }

    file.close();
}

void FilmMaster2000::reverse(Video& video) {
    size_t frameSize = video.channels * video.height * video.width;
    const size_t batchSize = 4;   // Batch size could be adjusted based on hardware

    size_t numPairs = static_cast<size_t>(video.numFrames) / 2;
    for (size_t batchStart = 0; batchStart < numPairs; batchStart += batchSize) {
        size_t batchEnd = std::min(batchStart + batchSize, numPairs);

        for (size_t i = batchStart; i < batchEnd; ++i) {
            uint8_t* frontFrame = video.frames[i].data.get();
            uint8_t* backFrame = video.frames[video.numFrames - 1 - i].data.get();

            for (size_t j = 0; j < frameSize; ++j) {
                std::swap(frontFrame[j], backFrame[j]);
            }
        }
    }
}

void FilmMaster2000::reverse_speed(Video& video) {
    size_t frameSize = video.channels * video.height * video.width;
    const size_t threadingThreshold = 100;
    const size_t batchSize = 4;

    // Lambda function to swap frame data
    auto simd_swap = [&](uint8_t* a, uint8_t* b) {
        size_t simdWidth = sizeof(__m256i);
        size_t alignedSize = frameSize / simdWidth * simdWidth;    // Calculate the aligned size for SIMD processing

        for (size_t i = 0; i < alignedSize; i += simdWidth) {
            __m256i va = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i));
            __m256i vb = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b + i));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + i), vb);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(b + i), va);
        }

        // Process remaining bytes that doesn't fit into SIMD registers
        for (size_t i = alignedSize; i < frameSize; ++i) {
            std::swap(a[i], b[i]);
        }
    };

    size_t numPairs = static_cast<size_t>(video.numFrames) / 2;

    if (static_cast<size_t>(video.numFrames) <= threadingThreshold) {
        // Single-threaded version for small workloads
        for (size_t batchStart = 0; batchStart < numPairs; batchStart += batchSize) {
            size_t batchEnd = std::min(batchStart + batchSize, numPairs);

            for (size_t i = batchStart; i < batchEnd; ++i) {
                simd_swap(video.frames[i].data.get(),
                          video.frames[video.numFrames - 1 - i].data.get());
            }
        }
    } else {
        // Multithreaded version for larger workloads
        unsigned int numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 2;

        size_t pairsPerThread = numPairs / numThreads;
        size_t remainingPairs = numPairs % numThreads;

        std::vector<std::thread> threads;   // Vector to hold the threads

        // Lambda function to reverse a batch of frames
        auto reverse_task = [&](size_t start, size_t end) {
            for (size_t batchStart = start; batchStart < end; batchStart += batchSize) {
                size_t batchEnd = std::min(batchStart + batchSize, end);

                for (size_t i = batchStart; i < batchEnd; ++i) {
                    simd_swap(video.frames[i].data.get(),
                              video.frames[video.numFrames - 1 - i].data.get());
                }
            }
        };

        size_t start = 0;
        for (unsigned int t = 0; t < numThreads; ++t) {
            size_t end = start + pairsPerThread + (t < remainingPairs ? 1 : 0);
            threads.emplace_back(reverse_task, start, end);
            start = end;
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }
}

void FilmMaster2000::reverse_memory(Video& video) {
    size_t frameSize = video.channels * video.height * video.width;
    for (size_t i = 0; i < static_cast<size_t>(video.numFrames) / 2; ++i) {
        // Swap the data of the current frame with its corresponding frame from the end
        for (size_t j = 0; j < frameSize; ++j) {
            std::swap(video.frames[i].data[j], video.frames[video.numFrames - 1 - i].data[j]);
        }
    }
}

void FilmMaster2000::clip_channel(Video& video, uint8_t channel, uint8_t min_val, uint8_t max_val) {
    size_t channelOffset = channel * video.height * video.width;
    size_t channelSize = video.height * video.width;

    const size_t batchSize = 4;

    for (size_t batchStart = 0; batchStart < static_cast<size_t>(video.numFrames); batchStart += batchSize) {
        size_t batchEnd = std::min(batchStart + batchSize, static_cast<size_t>(video.numFrames));

        for (size_t i = batchStart; i < batchEnd; ++i) {
            uint8_t* data = video.frames[i].data.get() + channelOffset;
            for (size_t j = 0; j < channelSize; ++j) {
                data[j] = std::clamp(data[j], min_val, max_val);   // Clamp value of data[j] between the range
            }
        }
    }
}

void FilmMaster2000::clip_channel_speed(Video& video, uint8_t channel, uint8_t min_val, uint8_t max_val) {
    size_t channelOffset = channel * video.height * video.width;
    size_t channelSize = video.height * video.width;

    const size_t threadingThreshold = 100;
    const size_t batchSize = 4;

    if (static_cast<size_t>(video.numFrames) <= threadingThreshold) {
        // Single-threaded fallback for small workloads
        for (size_t batchStart = 0; batchStart < static_cast<size_t>(video.numFrames); batchStart += batchSize) {
            size_t batchEnd = std::min(batchStart + batchSize, static_cast<size_t>(video.numFrames));
            for (size_t i = batchStart; i < batchEnd; ++i) {
                uint8_t* data = video.frames[i].data.get() + channelOffset;
                for (size_t j = 0; j < channelSize; ++j) {
                    data[j] = std::clamp(data[j], min_val, max_val);
                }
            }
        }
        return;
    }

    // Multithreaded version for larger workloads
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 2;

    std::mutex queueMutex;    // synchronize access to the task queue
    std::condition_variable condition;    // Notify worker threads when new tasks are available
    std::queue<std::pair<size_t, size_t>> tasks;   // To distribute tasks among worker threads
    std::vector<std::thread> threads;   // Vector to hold the worker threads
    bool stop = false;

    // Worker thread function
    auto worker = [&]() {
        while (true) {
            std::pair<size_t, size_t> task;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                condition.wait(lock, [&]() { return stop || !tasks.empty(); });

                if (stop && tasks.empty()) return;

                task = tasks.front();
                tasks.pop();
            }

            for (size_t i = task.first; i < task.second; ++i) {
                uint8_t* data = video.frames[i].data.get() + channelOffset;

#ifdef __AVX2__
                size_t simdWidth = sizeof(__m256i);
                size_t alignedSize = channelSize / simdWidth * simdWidth;

                __m256i minVec = _mm256_set1_epi8(min_val);
                __m256i maxVec = _mm256_set1_epi8(max_val);

                for (size_t j = 0; j < alignedSize; j += simdWidth) {
                    __m256i pixels = _mm256_loadu_si256(reinterpret_cast<__m256i*>(data + j));
                    __m256i clamped = _mm256_min_epu8(maxVec, _mm256_max_epu8(minVec, pixels));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(data + j), clamped);
                }

                for (size_t j = alignedSize; j < channelSize; ++j) {
                    data[j] = std::clamp(data[j], min_val, max_val);
                }
#else
                for (size_t j = 0; j < channelSize; ++j) {
                    data[j] = std::clamp(data[j], min_val, max_val);
                }
#endif
            }
        }
    };

    // Create worker threads
    for (unsigned int t = 0; t < numThreads; ++t) {
        threads.emplace_back(worker);
    }

    // Enqueue tasks in batches
    for (size_t batchStart = 0; batchStart < static_cast<size_t>(video.numFrames); batchStart += batchSize) {
        size_t batchEnd = std::min(batchStart + batchSize, static_cast<size_t>(video.numFrames));
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            tasks.emplace(batchStart, batchEnd);
        }
        condition.notify_one();
    }

    // Wait for all threads to finish
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    condition.notify_all();

    for (std::thread& thread : threads) {
        thread.join();
    }
}

void FilmMaster2000::clip_channel_memory(Video& video, uint8_t channel, uint8_t min_val, uint8_t max_val) {
    size_t channelOffset = channel * video.height * video.width;
    for (auto& frame : video.frames) {
        for (size_t h = 0; h < video.height; ++h) {
            for (size_t w = 0; w < video.width; ++w) {
                size_t index = channelOffset + h * video.width + w;
                frame.data[index] = std::clamp(frame.data[index], min_val, max_val);
            }
        }
    }
}

void FilmMaster2000::swap_channel(Video& video, uint8_t channel1, uint8_t channel2) {
    size_t channel1Offset = channel1 * video.height * video.width;
    size_t channel2Offset = channel2 * video.height * video.width;
    size_t channelSize = video.height * video.width;

    size_t batchSize = 4;

    for (size_t batchStart = 0; batchStart < static_cast<size_t>(video.numFrames); batchStart += batchSize) {
        size_t batchEnd = std::min(batchStart + batchSize, static_cast<size_t>(video.numFrames));

        // Process frames in batches
        for (size_t i = batchStart; i < batchEnd; ++i) {
            auto& frame = video.frames[i];

#ifdef __AVX2__
            size_t simdWidth = sizeof(__m256i);
            size_t alignedSize = channelSize / simdWidth * simdWidth;

            for (size_t j = 0; j < alignedSize; j += simdWidth) {
                __m256i data1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(frame.data.get() + channel1Offset + j));
                __m256i data2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(frame.data.get() + channel2Offset + j));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(frame.data.get() + channel1Offset + j), data2);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(frame.data.get() + channel2Offset + j), data1);
            }

            // Handle remaining elements that don't fit into SIMD registers
            for (size_t j = alignedSize; j < channelSize; ++j) {
                std::swap(frame.data[channel1Offset + j], frame.data[channel2Offset + j]);
            }
#else
            // Fallback for systems without AVX2
            for (size_t j = 0; j < channelSize; ++j) {
                std::swap(frame.data[channel1Offset + j], frame.data[channel2Offset + j]);
            }
#endif
        }
    }
}

void FilmMaster2000::swap_channel_speed(Video& video, uint8_t channel1, uint8_t channel2) {
    size_t channel1Offset = channel1 * video.height * video.width;
    size_t channel2Offset = channel2 * video.height * video.width;
    size_t channelSize = video.height * video.width;

    const size_t threadingThreshold = 100;
    const size_t batchSize = 4;

    if (static_cast<size_t>(video.numFrames) <= threadingThreshold) {
        // Single-threaded version for small workloads
        for (size_t batchStart = 0; batchStart < static_cast<size_t>(video.numFrames); batchStart += batchSize) {
            size_t batchEnd = std::min(batchStart + batchSize, static_cast<size_t>(video.numFrames));
            for (size_t i = batchStart; i < batchEnd; ++i) {
                auto& frame = video.frames[i];
#ifdef __AVX2__
                size_t simdWidth = sizeof(__m256i);
                size_t alignedSize = channelSize / simdWidth * simdWidth;

                for (size_t j = 0; j < alignedSize; j += simdWidth) {
                    __m256i data1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(frame.data.get() + channel1Offset + j));
                    __m256i data2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(frame.data.get() + channel2Offset + j));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(frame.data.get() + channel1Offset + j), data2);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(frame.data.get() + channel2Offset + j), data1);
                }

                for (size_t j = alignedSize; j < channelSize; ++j) {
                    std::swap(frame.data[channel1Offset + j], frame.data[channel2Offset + j]);
                }
#else
                for (size_t j = 0; j < channelSize; ++j) {
                    std::swap(frame.data[channel1Offset + j], frame.data[channel2Offset + j]);
                }
#endif
            }
        }
    } else {
        // Multithreaded version for larger workloads
        unsigned int numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 2;

        size_t framesPerThread = static_cast<size_t>(video.numFrames) / numThreads;
        size_t remainingFrames = static_cast<size_t>(video.numFrames) % numThreads;

        std::vector<std::thread> threads;

        auto swap_task = [&](size_t start, size_t end) {
            for (size_t batchStart = start; batchStart < end; batchStart += batchSize) {
                size_t batchEnd = std::min(batchStart + batchSize, end);
                for (size_t i = batchStart; i < batchEnd; ++i) {
                    auto& frame = video.frames[i];
#ifdef __AVX2__
                    size_t simdWidth = sizeof(__m256i);
                    size_t alignedSize = channelSize / simdWidth * simdWidth;

                    for (size_t j = 0; j < alignedSize; j += simdWidth) {
                        __m256i data1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(frame.data.get() + channel1Offset + j));
                        __m256i data2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(frame.data.get() + channel2Offset + j));
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(frame.data.get() + channel1Offset + j), data2);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(frame.data.get() + channel2Offset + j), data1);
                    }

                    for (size_t j = alignedSize; j < channelSize; ++j) {
                        std::swap(frame.data[channel1Offset + j], frame.data[channel2Offset + j]);
                    }
#else
                    for (size_t j = 0; j < channelSize; ++j) {
                        std::swap(frame.data[channel1Offset + j], frame.data[channel2Offset + j]);
                    }
#endif
                }
            }
        };

        size_t start = 0;
        for (unsigned int t = 0; t < numThreads; ++t) {
            size_t end = start + framesPerThread + (t < remainingFrames ? 1 : 0);
            threads.emplace_back(swap_task, start, end);
            start = end;
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }
}

void FilmMaster2000::swap_channel_memory(Video& video, uint8_t channel1, uint8_t channel2) {
    size_t channel1Offset = channel1 * video.height * video.width;
    size_t channel2Offset = channel2 * video.height * video.width;
    size_t channelSize = video.height * video.width;

    for (auto& frame : video.frames) {
        for (size_t i = 0; i < channelSize; ++i) {
            // Direct in-place swap to minimize memory overhead
            std::swap(frame.data[channel1Offset + i], frame.data[channel2Offset + i]);
        }
    }
}

void FilmMaster2000::scale_channel(Video& video, uint8_t channel, float scale_factor) {
    size_t channelOffset = channel * video.height * video.width;
    size_t channelSize = video.height * video.width;

    const size_t batchSize = 4;

    for (size_t batchStart = 0; batchStart < static_cast<size_t>(video.numFrames); batchStart += batchSize) {
        size_t batchEnd = std::min(batchStart + batchSize, static_cast<size_t>(video.numFrames));

        for (size_t i = batchStart; i < batchEnd; ++i) {
            uint8_t* data = video.frames[i].data.get() + channelOffset;

            for (size_t j = 0; j < channelSize; ++j) {
                int clampedValue = std::clamp(static_cast<int>(data[j]), 0, 255);
                int scaledValue = static_cast<int>(clampedValue * scale_factor);
                data[j] = std::clamp(scaledValue, 0, 255);
            }
        }
    }
}

void FilmMaster2000::scale_channel_speed(Video& video, uint8_t channel, float scale_factor) {
    size_t channelOffset = channel * video.height * video.width;
    size_t channelSize = video.height * video.width;

    const size_t threadingThreshold = 20;
    const size_t batchSize = 4;

    if (static_cast<size_t>(video.numFrames) <= threadingThreshold) {
        // Single-threaded fallback for small workloads
        for (size_t batchStart = 0; batchStart < static_cast<size_t>(video.numFrames); batchStart += batchSize) {
            size_t batchEnd = std::min(batchStart + batchSize, static_cast<size_t>(video.numFrames));

            for (size_t i = batchStart; i < batchEnd; ++i) {
                uint8_t* data = video.frames[i].data.get() + channelOffset;

                for (size_t j = 0; j < channelSize; ++j) {
                    int clampedValue = std::clamp(static_cast<int>(data[j]), 0, 255);
                    int scaledValue = static_cast<int>(clampedValue * scale_factor);
                    data[j] = std::clamp(scaledValue, 0, 255);
                }
            }
        }
        return;
    }

    // Multithreaded version for larger workloads
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 2;

    std::mutex queueMutex;
    std::condition_variable condition;
    std::queue<std::pair<size_t, size_t>> tasks;
    std::vector<std::thread> threads;
    bool stop = false;

    // Worker thread function
    auto worker = [&]() {
        while (true) {
            std::pair<size_t, size_t> task;

            {
                std::unique_lock<std::mutex> lock(queueMutex);
                condition.wait(lock, [&]() { return stop || !tasks.empty(); });

                if (stop && tasks.empty()) return;

                task = tasks.front();
                tasks.pop();
            }

            // Process the task
            for (size_t i = task.first; i < task.second; ++i) {
                uint8_t* data = video.frames[i].data.get() + channelOffset;

#ifdef __AVX2__
                size_t simdWidth = sizeof(__m256i);
                size_t alignedSize = channelSize / simdWidth * simdWidth;

                __m256i scaleVec = _mm256_set1_ps(scale_factor);
                __m256i minVec = _mm256_set1_epi32(0);
                __m256i maxVec = _mm256_set1_epi32(255);

                for (size_t j = 0; j < alignedSize; j += simdWidth) {
                    __m256i pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + j));
                    __m256 scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(pixels)), scaleVec);
                    __m256i clamped = _mm256_max_epi32(minVec, _mm256_min_epi32(maxVec, _mm256_cvtps_epi32(scaled)));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(data + j), clamped);
                }

                for (size_t j = alignedSize; j < channelSize; ++j) {
                    int clampedValue = std::clamp(static_cast<int>(data[j]), 0, 255);
                    int scaledValue = static_cast<int>(clampedValue * scale_factor);
                    data[j] = std::clamp(scaledValue, 0, 255);
                }
#else
                for (size_t j = 0; j < channelSize; ++j) {
                    int clampedValue = std::clamp(static_cast<int>(data[j]), 0, 255);
                    int scaledValue = static_cast<int>(clampedValue * scale_factor);
                    data[j] = std::clamp(scaledValue, 0, 255);
                }
#endif
            }
        }
    };

    // Create worker threads
    for (unsigned int t = 0; t < numThreads; ++t) {
        threads.emplace_back(worker);
    }

    // Enqueue tasks in batches
    for (size_t batchStart = 0; batchStart < static_cast<size_t>(video.numFrames); batchStart += batchSize) {
        size_t batchEnd = std::min(batchStart + batchSize, static_cast<size_t>(video.numFrames));
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            tasks.emplace(batchStart, batchEnd);
        }
        condition.notify_one();
    }

    // Wait for all threads to finish
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    condition.notify_all();

    for (std::thread& thread : threads) {
        thread.join();
    }
}

void FilmMaster2000::scale_channel_memory(Video& video, uint8_t channel, float scale_factor) {
    size_t channelOffset = channel * video.height * video.width;
    size_t channelSize = video.height * video.width;

    for (auto& frame : video.frames) {
        uint8_t* data = frame.data.get() + channelOffset;

        // Process in chunks
        const size_t chunkSize = 1024;
        for (size_t start = 0; start < channelSize; start += chunkSize) {
            size_t end = std::min(start + chunkSize, channelSize);

            for (size_t i = start; i < end; ++i) {
                int clampedValue = std::clamp(static_cast<int>(data[i]), 0, 255);
                int scaledValue = static_cast<int>(clampedValue * scale_factor);
                data[i] = std::clamp(scaledValue, 0, 255);
            }
        }
    }
}

void FilmMaster2000::grayscale(Video& video) {
    const size_t batchSize = 4;

    for (size_t batchStart = 0; batchStart < static_cast<size_t>(video.numFrames); batchStart += batchSize) {
        size_t batchEnd = std::min(batchStart + batchSize, static_cast<size_t>(video.numFrames));

        for (size_t i = batchStart; i < batchEnd; ++i) {
            auto& frame = video.frames[i];
            for (size_t h = 0; h < video.height; ++h) {
                for (size_t w = 0; w < video.width; ++w) {
                    size_t pixelIndex = h * video.width + w;
                    int sum = 0;
                    for (size_t ch = 0; ch < video.channels; ++ch) {
                        sum += frame.data[ch * video.height * video.width + pixelIndex];
                    }
                    uint8_t grayValue = sum / video.channels;
                    for (size_t ch = 0; ch < video.channels; ++ch) {
                        frame.data[ch * video.height * video.width + pixelIndex] = grayValue;
                    }
                }
            }
        }
    }
}

void FilmMaster2000::grayscale_speed(Video& video) {
    const size_t frameThreshold = 100;
    const size_t batchSize = 4;

    // Single-threaded processing for small workloads
    if (static_cast<size_t>(video.numFrames) <= frameThreshold) {
        for (auto& frame : video.frames) {
            uint8_t* data = frame.data.get();
#ifdef __AVX2__
            size_t simdWidth = sizeof(__m256i);
            size_t channelSize = video.height * video.width;
            size_t alignedSize = channelSize / simdWidth * simdWidth;

            for (size_t i = 0; i < alignedSize; i += simdWidth) {
                __m256i ch0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(data + i));
                __m256i ch1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(data + channelSize + i));
                __m256i ch2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(data + 2 * channelSize + i));

                // Average across channels
                __m256i sum = _mm256_add_epi8(ch0, _mm256_add_epi8(ch1, ch2));
                __m256i avg = _mm256_div_epu8(sum, _mm256_set1_epi8(video.channels));

                // Store the grayscale value back into all channels
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(data + i), avg);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(data + channelSize + i), avg);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(data + 2 * channelSize + i), avg);
            }

            // Scalar fallback for remaining pixels
            for (size_t i = alignedSize; i < channelSize; ++i) {
                int sum = 0;
                for (size_t ch = 0; ch < video.channels; ++ch) {
                    sum += data[ch * channelSize + i];
                }
                uint8_t grayValue = sum / video.channels;
                for (size_t ch = 0; ch < video.channels; ++ch) {
                    data[ch * channelSize + i] = grayValue;
                }
            }
#else
            // Scalar implementation if SIMD is not available
            for (size_t h = 0; h < video.height; ++h) {
                for (size_t w = 0; w < video.width; ++w) {
                    size_t pixelIndex = h * video.width + w;
                    int sum = 0;
                    for (size_t ch = 0; ch < video.channels; ++ch) {
                        sum += data[ch * video.height * video.width + pixelIndex];
                    }
                    uint8_t grayValue = sum / video.channels;
                    for (size_t ch = 0; ch < video.channels; ++ch) {
                        data[ch * video.height * video.width + pixelIndex] = grayValue;
                    }
                }
            }
#endif
        }
        return;
    }

    // Multithreaded processing for larger workloads
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 2;

    size_t framesPerThread = video.numFrames / numThreads;
    size_t remainingFrames = video.numFrames % numThreads;

    std::vector<std::thread> threads;
    auto process_frames = [&](size_t start, size_t end) {
        for (size_t batchStart = start; batchStart < end; batchStart += batchSize) {
            size_t batchEnd = std::min(batchStart + batchSize, end);
            for (size_t i = batchStart; i < batchEnd; ++i) {
                uint8_t* data = video.frames[i].data.get();
                for (size_t h = 0; h < video.height; ++h) {
                    for (size_t w = 0; w < video.width; ++w) {
                        size_t pixelIndex = h * video.width + w;
                        int sum = 0;
                        for (size_t ch = 0; ch < video.channels; ++ch) {
                            sum += data[ch * video.height * video.width + pixelIndex];
                        }
                        uint8_t grayValue = sum / video.channels;
                        for (size_t ch = 0; ch < video.channels; ++ch) {
                            data[ch * video.height * video.width + pixelIndex] = grayValue;
                        }
                    }
                }
            }
        }
    };

    // Launch threads
    size_t start = 0;
    for (unsigned int t = 0; t < numThreads; ++t) {
        size_t end = start + framesPerThread + (t < remainingFrames ? 1 : 0);
        threads.emplace_back(process_frames, start, end);
        start = end;
    }

    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }
}

void FilmMaster2000::grayscale_memory(Video& video) {
    for (auto& frame : video.frames) {
        for (size_t h = 0; h < video.height; ++h) {
            for (size_t w = 0; w < video.width; ++w) {
                size_t pixelIndex = h * video.width + w;
                int sum = 0;
                for (size_t ch = 0; ch < video.channels; ++ch) {
                    sum += frame.data[ch * video.height * video.width + pixelIndex];
                }
                uint8_t grayValue = sum / video.channels;
                for (size_t ch = 0; ch < video.channels; ++ch) {
                    frame.data[ch * video.height * video.width + pixelIndex] = grayValue;
                }
            }
        }
    }
}
