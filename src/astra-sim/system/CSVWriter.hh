/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <cstdint>
#include <fstream>
#include <sys/stat.h>
#include "sys_layer/containers/FixedList.hpp"
#include "sys_layer/containers/FixedString.hpp"
#include "sys_layer/containers/FixedPair.hpp"

namespace AstraSim {

/**
 * CSV文件写入器类
 */
class CSVWriter {
public:
    CUDA_HOST_DEVICE CSVWriter(custom::FixedString path, custom::FixedString name);
    CUDA_HOST_DEVICE ~CSVWriter() {
#ifndef __CUDA_ARCH__
        if (myFile.is_open()) {
            myFile.close();
        }
#endif
    }
    CUDA_HOST_DEVICE void initialize_csv(int rows, int cols);
    CUDA_HOST_DEVICE void finalize_csv(custom::FixedList<custom::FixedList<custom::FixedPair<uint64_t, double>, 32>, 32> dims);
    CUDA_HOST_DEVICE void write_cell(int row, int column, custom::FixedString data);
    
#ifndef __CUDA_ARCH__
    inline bool exists_test(const custom::FixedString& name) {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
    }
#endif

    std::fstream myFile;
    custom::FixedString name;
    custom::FixedString path;
};

} // namespace AstraSim
