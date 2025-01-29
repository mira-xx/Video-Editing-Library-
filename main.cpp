// Copyright 2024 <Mohana Maran>

#include "FilmMaster2000.h"
#include <iostream>
#include <malloc.h>

// Print usage instructions
void print_usage() {
    std::cout << "Usage: ./runme <input_file> <output_file> <flags> <function> [<args>...]\n";
    std::cout << "Flags:\n";
    std::cout << "  -S  Optimize for speed\n";
    std::cout << "  -M  Optimize for memory\n";
    std::cout << "Functions:\n";
    std::cout << "  reverse\n";
    std::cout << "  swap_channel <ch1,ch2>\n";
    std::cout << "  clip_channel <channel> [<min>,<max>]\n";
    std::cout << "  scale_channel <channel> <scale_factor>\n";
    std::cout << "  grayscale\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage();
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    std::string flag = argv[3];
    std::string function;
    bool optimizeForMemory = false;
    bool optimizeForSpeed = false;
    int functionArgIndex = 3;   // Default index for function arguments

    // Check for optimization flags
    if (flag == "-S") {
        optimizeForSpeed = true;
        if (argc < 5) {
            print_usage();
            return 1;
        }
        function = argv[4];  // When there is a flag, the function name is at index 4
        functionArgIndex = 4;
    } else if (flag == "-M") {
        optimizeForMemory = true;
        if (argc < 5) {
            print_usage();
            return 1;
        }
        function = argv[4];
        functionArgIndex = 4;
    } else {
        function = flag;  // When there is no flag, the function name is at index 3
    }

    try {
        Video video;
        if (optimizeForMemory) {
            video = FilmMaster2000::decode_memory(inputFile);
        } else if (optimizeForSpeed) {
            video = FilmMaster2000::decode_speed(inputFile);
        } else {
            video = FilmMaster2000::decode(inputFile);
        }

        // Reverse
        if (function == "reverse") {
            if (optimizeForMemory) {
                FilmMaster2000::reverse_memory(video);
            } else if (optimizeForSpeed) {
                FilmMaster2000::reverse_speed(video);
            } else {
                FilmMaster2000::reverse(video);
            }
            // Swap Channel
        } else if (function == "swap_channel") {
            if (argc < functionArgIndex + 2) {
                std::cerr << "Error: Missing arguments for swap_channel. Provide <ch1,ch2>.\n";
                print_usage();
                return 1;
            }
            std::string channels = argv[functionArgIndex + 1];
            size_t commaPos = channels.find(',');
            if (commaPos == std::string::npos) {
                std::cerr << "Error: Invalid format for swap_channel. Provide <ch1,ch2>.\n";
                print_usage();
                return 1;
            }
            unsigned char ch1 = static_cast<unsigned char>(std::stoi(channels.substr(0, commaPos)));
            unsigned char ch2 = static_cast<unsigned char>(std::stoi(channels.substr(commaPos + 1)));
            if (optimizeForMemory) {
                FilmMaster2000::swap_channel_memory(video, ch1, ch2);
            } else if (optimizeForSpeed) {
                FilmMaster2000::swap_channel_speed(video, ch1, ch2);
            } else {
                FilmMaster2000::swap_channel(video, ch1, ch2);
            }
            // Clip Channel
        } else if (function == "clip_channel") {
            if (argc < functionArgIndex + 3) {
                std::cerr << "Error: Missing arguments for clip_channel. Provide <channel> [<min>,<max>].\n";
                print_usage();
                return 1;
            }
            unsigned char channel = static_cast<unsigned char>(std::stoi(argv[functionArgIndex + 1]));
            std::string minMax = argv[functionArgIndex + 2];
            size_t commaPos = minMax.find(',');
            if (commaPos == std::string::npos) {
                std::cerr << "Error: Invalid format for clip_channel. Provide <channel> [<min>,<max>].\n";
                print_usage();
                return 1;
            }
            unsigned char min = static_cast<unsigned char>(std::stoi(minMax.substr(1, commaPos -1)));   // to skip '['
            unsigned char max = static_cast<unsigned char>(std::stoi(minMax.substr(commaPos + 1, minMax.length() - commaPos - 2)));   // to skip ',' and ']'
            if (optimizeForMemory) {
                FilmMaster2000::clip_channel_memory(video, channel, min, max);
            } else if (optimizeForSpeed) {
                FilmMaster2000::clip_channel_speed(video, channel, min, max);
            } else {
                FilmMaster2000::clip_channel(video, channel, min, max);
            }
            // Scale Channel
        } else if (function == "scale_channel") {
            if (argc < functionArgIndex + 3) {
                std::cerr << "Error: Missing arguments for scale_channel. Provide <channel> <scale_factor>.\n";
                print_usage();
                return 1;
            }
            unsigned char channel = static_cast<unsigned char>(std::stoi(argv[functionArgIndex + 1]));
            float scaleFactor = std::stof(argv[functionArgIndex + 2]);
            if (optimizeForMemory) {
                FilmMaster2000::scale_channel_memory(video, channel, scaleFactor);
            } else if (optimizeForSpeed) {
                FilmMaster2000::scale_channel_speed(video, channel, scaleFactor);
            } else {
                FilmMaster2000::scale_channel(video, channel, scaleFactor);
            }
            // Grayscale
        } else if (function == "grayscale") {
            if (optimizeForMemory) {
                FilmMaster2000::grayscale_memory(video);
            } else if (optimizeForSpeed) {
                FilmMaster2000::grayscale_speed(video);
            } else {
                FilmMaster2000::grayscale(video);
            }
        } else {
            std::cerr << "Error: Unknown function '" << function << "'.\n";
            print_usage();
            return 1;
        }
        if (optimizeForMemory) {
            FilmMaster2000::encode_memory(outputFile, video);
        } else if (optimizeForSpeed) {
            FilmMaster2000::encode_speed(outputFile, video);
        } else {
            FilmMaster2000::encode(outputFile, video);
        }

        video.frames.clear();
        video.frames.shrink_to_fit();
        video = Video();   // Reset the Video object
        std::cout << "Video processing completed successfully." << std::endl;
        malloc_trim(0);    // Force memory release
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

