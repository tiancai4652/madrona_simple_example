/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/CSVWriter.hh"
#include "astra-sim/system/Common.hh"

#include <cassert>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>

namespace AstraSim {

CUDA_HOST_DEVICE
CSVWriter::CSVWriter(custom::FixedString path, custom::FixedString name) {
    this->path = path;
    this->name = name;
#ifndef __CUDA_ARCH__
    std::cout << "Created CSVWriter with path=" << path.c_str() 
              << ", name=" << name.c_str() << std::endl;
#endif
}

CUDA_HOST_DEVICE
void CSVWriter::initialize_csv(int rows, int cols) {
#ifndef __CUDA_ARCH__
    std::cout << "CSV path and filename: " << (path + name).c_str() << std::endl;
    int trial = 10000;
    do {
        myFile.open((path + name).c_str(), std::fstream::out);
        trial--;
    } while (!myFile.is_open() && trial > 0);
    if (trial == 0) {
        std::cerr << "Unable to create file: " << path.c_str() << std::endl;
        std::cerr << "This error is fatal. Please make sure the CSV write path exists." << std::endl;
        exit(1);
    }
    do {
        myFile.close();
    } while (myFile.is_open());

    do {
        myFile.open((path + name).c_str(), std::fstream::out | std::fstream::in);
    } while (!myFile.is_open());

    if (!myFile) {
        std::cerr << "Unable to open file: " << path.c_str() << std::endl;
        std::cerr << "This error is fatal. Please make sure the CSV write path exists." << std::endl;
        exit(1);
    } else {
        std::cout << "Success in opening CSV file for writing the report." << std::endl;
    }

    myFile.seekp(0, std::ios_base::beg);
    myFile.seekg(0, std::ios_base::beg);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols - 1; j++) {
            myFile << ',';
        }
        myFile << '\n';
    }
    do {
        myFile.close();
    } while (myFile.is_open());
#endif
}

CUDA_HOST_DEVICE
void CSVWriter::finalize_csv(custom::FixedList<custom::FixedList<custom::FixedPair<uint64_t, double>, 32>, 32> dims) {
#ifndef __CUDA_ARCH__
    std::cout << "Path to create CSVs is: " << path.c_str() << std::endl;
    int trial = 10000;
    do {
        myFile.open((path + name).c_str(), std::fstream::out);
        trial--;
    } while (!myFile.is_open() && trial > 0);
    if (trial == 0) {
        std::cerr << "Unable to create file: " << (path + name).c_str() << std::endl;
        std::cerr << "This error is fatal. Please make sure the CSV write path exists." << std::endl;
        exit(1);
    }
    do {
        myFile.close();
    } while (myFile.is_open());

    do {
        myFile.open((path + name).c_str(), std::fstream::out | std::fstream::in);
    } while (!myFile.is_open());

    if (!myFile) {
        std::cerr << "Unable to open file: " << (path + name).c_str() << std::endl;
        std::cerr << "This error is fatal. Please make sure the CSV write path exists." << std::endl;
        exit(1);
    } else {
        std::cout << "Success in opening file" << std::endl;
    }

    myFile.seekp(0, std::ios_base::beg);
    myFile.seekg(0, std::ios_base::beg);
    custom::FixedVector<typename custom::FixedList<custom::FixedPair<uint64_t, double>, 32>::iterator, 32> dims_it;
    custom::FixedVector<typename custom::FixedList<custom::FixedPair<uint64_t, double>, 32>::iterator, 32> dims_it_end;
    
    for (auto& dim : dims) {
        dims_it.push_back(dim.begin());
        dims_it_end.push_back(dim.end());
    }

    int dim_num = 1;
    myFile << " time (us) ";
    myFile << ",";
    for (uint64_t i = 0; i < dims.size(); i++) {
        myFile << "dim" + std::to_string(dim_num) + " util";
        myFile << ',';
        dim_num++;
    }
    myFile << '\n';

    while (true) {
        uint64_t finished = 0;
        uint64_t compare;
        for (uint64_t i = 0; i < dims_it.size(); i++) {
            if (dims_it[i] != dims_it_end[i]) {
                if (i == 0) {
                    myFile << std::to_string((*dims_it[i]).first / FREQ);
                    myFile << ",";
                    compare = (*dims_it[i]).first;
                } else {
                    assert(compare == (*dims_it[i]).first);
                }
            }
            if (dims_it[i] == dims_it_end[i]) {
                finished++;
                myFile << ",";
                continue;
            } else {
                myFile << std::to_string((*dims_it[i]).second);
                myFile << ',';
                ++dims_it[i];
            }
        }
        myFile << '\n';
        if (finished == dims_it_end.size()) {
            break;
        }
    }
    myFile.close();
#endif
}

CUDA_HOST_DEVICE
void CSVWriter::write_cell(int row, int column, custom::FixedString data) {
#ifndef __CUDA_ARCH__
    custom::FixedString str;
    custom::FixedString tmp;

    int status = 1;
    int fildes = -1;
    do {
        fildes = open((path + name).c_str(), O_RDWR);
    } while (fildes == -1);

    do {
        status = lockf(fildes, F_TLOCK, (off_t)1000000);
    } while (status != 0);

    char buf[1];
    while (row > 0) {
        status = read(fildes, buf, 1);
        str = str + (*buf);
        if (*buf == '\n') {
            row--;
        }
    }
    while (column > 0) {
        status = read(fildes, buf, 1);
        str = str + (*buf);
        if (*buf == ',') {
            column--;
        }
        if (*buf == '\n') {
            std::cerr << "Fatal error in inserting cell!" << std::endl;
            exit(1);
        }
    }
    str = str + data;
    while (read(fildes, buf, 1)) {
        str = str + (*buf);
    }

    do {
        lseek(fildes, 0, SEEK_SET);
        status = write(fildes, str.c_str(), str.length());
    } while (status != str.length());

    do {
        status = lseek(fildes, 0, SEEK_SET);
    } while (status == -1);

    do {
        status = lockf(fildes, F_ULOCK, (off_t)1000000);
    } while (status != 0);

    do {
        status = close(fildes);
    } while (status == -1);
#endif
}

} // namespace AstraSim
