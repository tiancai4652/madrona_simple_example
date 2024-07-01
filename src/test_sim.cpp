#include <iostream>
#include <vector>

// Test case 1: Empty input
void testEmptyInput() {
    PktBuf data;
    insertionSort(data);
    // No assertion, just checking if the function runs without errors
}

// Test case 2: Already sorted input
void testSortedInput() {
    PktBuf data;
    // Add some packets in ascending order of enqueue_time
    for (int i = 0; i < 10; i++) {
        Pkt pkt;
        pkt.enqueue_time = i;
        data.push_back(pkt);
    }
    insertionSort(data);
    // No assertion, just checking if the function runs without errors
}

// Test case 3: Reverse sorted input
void testReverseSortedInput() {
    PktBuf data;
    // Add some packets in descending order of enqueue_time
    for (int i = 9; i >= 0; i--) {
        Pkt pkt;
        pkt.enqueue_time = i;
        data.push_back(pkt);
    }
    insertionSort(data);
    // No assertion, just checking if the function runs without errors
}

// Test case 4: Random input
void testRandomInput() {
    PktBuf data;
    // Add some packets with random enqueue_time values
    for (int i = 0; i < 10; i++) {
        Pkt pkt;
        pkt.enqueue_time = rand() % 100;
        data.push_back(pkt);
    }
    insertionSort(data);
    // No assertion, just checking if the function runs without errors
}

int main() {
    testEmptyInput();
    testSortedInput();
    testReverseSortedInput();
    testRandomInput();

    std::cout << "All tests passed!" << std::endl;

    return 0;
}