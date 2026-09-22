#include "sim.hpp"
#include "sim_debug.hpp"

#include <limits>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {
void Sim::resetNetworkState()
{
    now = 0.0;
    nextPortID = 0;
    numTopoNodes = 0;
    nodeLookupBase = 0;
    nodeLookupSpan = 0;
    numTopoLinks = 0;
    numPorts = 0;
    numFlowMetaEntities = 0;
    numNpus = 0;
    enableBuffer = 1;
    enablePfc = 0;
    pfcEgress = 0;
    defaultLinkDelay = 0.001;
    propagationInterval = 0.0;
    pfcXoffThreshold = 1e9;
    pfcXonThreshold = 0.5e9;
    dtMin = 0.0;
    qosMode = 0;
    systemLogStep = 0;

    // #pragma unroll 1: these loops run over compile-time-constant bounds
    // (up to 1,048,576 iterations). NVRTC -O3 fully unrolls them, producing
    // megabytes of PTX and a ~400s compile for this TU. The pragma keeps
    // them as loops; runtime cost is memory-bound and unchanged.
    for (int32_t i = 0; i < PFC_MAX_PRIORITY; i++) {
        priorWeights[i] = 0.0;
    }

#pragma unroll 1
    for (int32_t i = 0; i < MAX_TOPO_NODES; i++) {
        topoNodes[i] = TopoNodeState {};
        nodeSlotLookup[i] = -1;
    }

#pragma unroll 1
    for (int32_t i = 0; i < MAX_TOPO_LINKS; i++) {
        topoLinks[i] = TopoLinkState {};
    }

#pragma unroll 1
    for (int32_t i = 0; i < MAX_FLOW_META_LOOKUP; i++) {
        flowMetaLookupIds[i] = -1;
        flowMetaEntityLookup[i] = Entity::none();
    }

#pragma unroll 1
    for (int32_t i = 0; i < MAX_FLOWS; i++) {
        flowMetaEntities[i] = Entity::none();
    }

#pragma unroll 1
    for (uint32_t i = 0; i < MAX_NPUS; i++) {
        npuEntities[i] = Entity::none();
    }
}

}
