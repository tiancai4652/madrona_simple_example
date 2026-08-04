#pragma once

// Compatibility switches for the migrated system layer. Production builds
// always use the real network bridge and keep verbose source diagnostics off.
#define DEV_MODE 2
#define SIMPLE_LOG_MODE 0
#define SYS_LOG 0
#define SYS_LOG_SPECIAL 0
#define SYS_LOG_TARGET_NODE 0
#define SYS_LOG_ENABLED_LIMIT_COMM_SIZE_DURATION 0
#define SYS_LOG_LIMIT_COMM_SIZE 1000
#define TEST_RING_TOPO_LOG_FOR_CPU_ONLY 0
#define TEST_RING_COLLECTIVE_COMM_LOG_FOR_CPU_ONLY 0
#define TEST_CHAKRA_NODE_LOG_FOR_CPU_ONLY 0
#define TEST_ALL_RING_COLLECTIVE_COMM_LOG_FOR_CPU_ONLY 0
