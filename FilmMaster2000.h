// Copyright 2024 <Mohana Maran>

#ifndef FILMMASTER2000_H
#define FILMMASTER2000_H

#include <vector>
#include <string>
#include <cstdint>
#include <memory>
#include <fstream>
#include <algorithm>

struct Frame {
    std::unique_ptr<uint8_t[]> data;   // Flat array for channel, height, and width
};

struct Video {
    long numFrames;   // // NOLINT(runtime/int) - Number of Frames in the full video must be long for this project
    uint8_t channels;
    uint8_t height;
    uint8_t width;
    std::vector<Frame> frames;
};

class FilmMaster2000 {
    public:
        static Video decode(const std::string& filename);
        static Video decode_speed(const std::string& filename);
        static Video decode_memory(const std::string& filename);

        static void encode(const std::string& filename, const Video& video);
        static void encode_speed(const std::string& filename, const Video& video);
        static void encode_memory(const std::string& filename, Video& video);

        static void reverse(Video& video);
        static void reverse_speed(Video& video);
        static void reverse_memory(Video& video);

        static void clip_channel(Video& video, uint8_t channel, uint8_t min_val, uint8_t max_val);
        static void clip_channel_speed(Video& video, uint8_t channel, uint8_t min_val, uint8_t max_val);
        static void clip_channel_memory(Video& video, uint8_t channel, uint8_t min_val, uint8_t max_val);

        static void swap_channel(Video& video, uint8_t channel1, uint8_t channel2);
        static void swap_channel_speed(Video& video, uint8_t channel1, uint8_t channel2);
        static void swap_channel_memory(Video& video, uint8_t channel1, uint8_t channel2);

        static void scale_channel(Video& video, uint8_t channel, float scale_factor);
        static void scale_channel_speed(Video& video, uint8_t channel, float scale_factor);
        static void scale_channel_memory(Video& video, uint8_t channel, float scale_factor);

        static void grayscale(Video& video);
        static void grayscale_speed(Video& video);
        static void grayscale_memory(Video& video);
};

#endif