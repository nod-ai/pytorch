#include <torch/csrc/distributed/c10d/intra_node_comm.hpp>

#include <ATen/zoom/ZoomContext.h>
#include <c10/zoom/ZoomGuard.h>
#include <c10/util/Logging.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <iostream>
#include <utility>

#include <fcntl.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <hip/hip_runtime.h>

namespace c10d::intra_node_comm {

static std::vector<std::string> ENABLE_INTRA_NODE_COMM = {
    "ENABLE_INTRA_NODE_COMM"};
// Forces detectedTopology() to return Topology::FULLY_CONNECTED, so
// IntraNodeComm can be used even without NVLink connection. This is only used
// for testing purposes.
static std::vector<std::string> TEST_INTRA_NODE_COMM = {"TEST_INTRA_NODE_COMM"};

////////////////////////////////////////////////////////////////////////////////
// HIP Functions
////////////////////////////////////////////////////////////////////////////////

bool isIntraNodeCommSupported();

std::optional<HybridCubeMesh> getHybridCubeMesh(NvlMesh nvlMesh);

void* initP2pState();

void* initTopoInfo(Topology topology, NvlMesh nvlMesh, size_t rank);

////////////////////////////////////////////////////////////////////////////////
// Topology Detection
////////////////////////////////////////////////////////////////////////////////

static std::ostream& operator<<(std::ostream& os, const NvlMesh& nvlMesh) {
  std::ostringstream oss;
  for (size_t i = 0; i < kMaxDevices; ++i) {
    for (size_t j = 0; j < kMaxDevices; ++j) {
      oss << nvlMesh[i][j] << " ";
    }
    oss << '\n';
  }
  os << oss.str();
  return os;
}

static bool isSame(NvlMesh lhs, NvlMesh rhs) {
  for (size_t i = 0; i < kMaxDevices; ++i) {
    for (size_t j = 0; j < kMaxDevices; ++j) {
      if (lhs[i][j] != rhs[i][j]) {
        return false;
      }
    }
  }
  return true;
}

/**
 * Query the nvlink connection among devices.
 */
static NvlMesh getNvlMesh(const std::vector<std::string>& rankToBusId) {
  return {};
}

/**
 * Determine if the devices form a hybrid cube mesh
 * topology given a NvlMesh.
 */
static bool isHybridCubeMesh(const NvlMesh nvlMesh) {
  std::array<size_t, kMaxDevices> numNeighbors = {};
  for (size_t i = 0; i < kMaxDevices; ++i) {
    for (size_t j = 0; j < kMaxDevices; ++j) {
      if (nvlMesh[i][j] > 0) {
        numNeighbors[i] += 1;
      }
    }
  }
  for (size_t i = 0; i < kMaxDevices; ++i) {
    // TODO: this is insufficent and needs revisit
    if (numNeighbors[i] != 4) {
      return false;
    }
  }
  return true;
}

/**
 * Detech topology given a NvlMesh.
 */
static Topology detectTopology(const NvlMesh nvlMesh, size_t worldSize) {
  if (getCvarBool(TEST_INTRA_NODE_COMM, false)) {
    return Topology::FULLY_CONNECTED;
  }
  bool fullyConnected = true;
  for (size_t i = 0; i < worldSize - 1; ++i) {
    for (size_t j = i + 1; j < worldSize; ++j) {
      if (nvlMesh[i][j] == 0 || nvlMesh[j][i] == 0) {
        fullyConnected = false;
      }
    }
  }
  if (fullyConnected) {
    LOG(INFO) << "IntraNodeComm: Topology::FULLY_CONNECTED";
    return Topology::FULLY_CONNECTED;
  }
  if (worldSize == kMaxDevices && getHybridCubeMesh(nvlMesh) != std::nullopt) {
    LOG(INFO) << "IntraNodeComm: Topology::HYBRID_CUBE_MESH";
    return Topology::HYBRID_CUBE_MESH;
  }
  LOG(INFO) << "IntraNodeComm: Topology::UNKNOWN";
  return Topology::UNKNOWN;
};

////////////////////////////////////////////////////////////////////////////////
// Rendezvous and Initialization
////////////////////////////////////////////////////////////////////////////////

IntraNodeComm::IntraNodeComm(
    c10::intrusive_ptr<c10d::Store> store,
    size_t rank,
    size_t worldSize,
    std::optional<size_t> bufferSize)
    : store_(std::move(store)),
      rank_(rank),
      worldSize_(worldSize),
      bufferSize_(bufferSize.has_value() ? *bufferSize : kDefaultBufferSize) {
  rendezvous();
}

IntraNodeComm::~IntraNodeComm() {
  if (!isInitialized_) {
    return;
  }
  // Intentionally releasing resources without synchronizing devices. The
  // teardown logic is safe for propoerly sync'd user program. We don't want
  // improperly sync'd user program to hang here.
  for (size_t r = 0; r < worldSize_; ++r) {
    if (r == rank_) {
      continue;
    }
    C10_ZOOM_CHECK(hipIpcCloseMemHandle(p2pStates_[r]));
    C10_ZOOM_CHECK(hipIpcCloseMemHandle(buffers_[r]));
  }
  C10_ZOOM_CHECK(hipFree(p2pStates_[rank_]));
  C10_ZOOM_CHECK(hipFree(buffers_[rank_]));
  if (topoInfo_ != nullptr) {
    C10_ZOOM_CHECK(hipFree(topoInfo_));
  }
  C10_ZOOM_CHECK(hipFree(p2pStatesDev_));
  C10_ZOOM_CHECK(hipFree(buffersDev_));
}

bool IntraNodeComm::isEnabled() {
  return getCvarBool(ENABLE_INTRA_NODE_COMM, false);
}

/**
 * Use c10d::Store to perform allgather on a trivially copyable type.
 */
template <typename T>
std::vector<T> storeAllGather(
    const c10::intrusive_ptr<c10d::Store>& store,
    const std::string& prefix,
    size_t rank,
    size_t worldSize,
    T val) {
  static_assert(std::is_trivially_copyable_v<T>);

  std::vector<std::string> peerKeys;
  for (size_t r = 0; r < worldSize; ++r) {
    std::ostringstream oss;
    oss << prefix << "-" << r;
    peerKeys.push_back(oss.str());
  }

  {
    std::vector<uint8_t> payload(
        reinterpret_cast<uint8_t*>(&val),
        reinterpret_cast<uint8_t*>(&val) + sizeof(T));
    store->set(peerKeys[rank], payload);
  }

  std::vector<T> peerVals;
  for (size_t r = 0; r < worldSize; ++r) {
    if (r == rank) {
      peerVals.push_back(val);
      continue;
    }
    store->wait({peerKeys[r]});
    auto payload = store->get(peerKeys[r]);
    TORCH_CHECK(payload.size() == sizeof(T));
    T peerVal{};
    std::memcpy(&peerVal, payload.data(), sizeof(T));
    peerVals.push_back(peerVal);
  }
  return peerVals;
}

bool IntraNodeComm::rendezvous() {
  if (isInitialized_) {
    return true;
  }
  return false;
}

} // namespace c10d::intra_node_comm
