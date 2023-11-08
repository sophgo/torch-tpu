#include <c10/util/Exception.h>
#include <c10/core/Device.h>
#include <ATen/Utils.h>

#include <vector>
#include <deque>

#include "torch_tpu/csrc/aten/TPUGeneratorImpl.h"
#include "torch_tpu/csrc/core/TPUDeviceManager.h"

namespace at_tpu{

namespace detail {
static std::once_flag num_tpu_init_flag;
static int64_t num_tpus;
static std::deque<std::once_flag> tpu_gens_init_flag;
static std::vector<at::Generator> default_gens_tpu;

static void initTPUGenVector() {
    num_tpus = tpu::TPUGetDeviceCount();
    tpu_gens_init_flag.resize(num_tpus);
    default_gens_tpu.resize(num_tpus);
}

const at::Generator& getDefaultTPUGenerator(c10::DeviceIndex device_index) {
  std::call_once(num_tpu_init_flag, initTPUGenVector);
  c10::DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = tpu::TPUGetDeviceIndex();
  } else {
    TORCH_CHECK(idx >= 0 && idx < num_tpus);
  }
  std::call_once(tpu_gens_init_flag[idx], [&] {
    default_gens_tpu[idx] = at::make_generator<TPUGeneratorImpl>(idx);
    default_gens_tpu[idx].seed();
  });
  return default_gens_tpu[idx];
}

at::Generator createTPUGenerator(c10::DeviceIndex device_index) {
  std::call_once(num_tpu_init_flag, initTPUGenVector);
  c10::DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = tpu::TPUGetDeviceIndex();
  }
  TORCH_CHECK(idx >= 0 && idx < num_tpus, "The device_index is invalid.");
  auto gen = at::make_generator<TPUGeneratorImpl>(idx);
  auto tpu_gen = at::check_generator<TPUGeneratorImpl>(gen);
  tpu_gen->set_current_seed(c10::default_rng_seed_val);
  tpu_gen->set_philox_offset_per_thread(0);
  return gen;
}


}; // namespace detail

TPUGeneratorImpl::TPUGeneratorImpl(c10::DeviceIndex device_index)
    : c10::GeneratorImpl{ c10::Device(c10::DeviceType::PrivateUse1, device_index),
                          c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)}
{
}

void TPUGeneratorImpl::set_current_seed(uint64_t seed) {
    seed_ = seed;
    philox_offset_per_thread_ = 0;
}

uint64_t TPUGeneratorImpl::current_seed() const {
    return seed_;
}

uint64_t TPUGeneratorImpl::seed() {
    auto random = c10::detail::getNonDeterministicRandom(true);
    this->set_current_seed(random);
    return random;
}

uint64_t TPUGeneratorImpl::philox_offset_per_thread() const {
    return philox_offset_per_thread_;
}

void TPUGeneratorImpl::set_philox_offset_per_thread(uint64_t offset) {
    TORCH_CHECK(offset % 4 == 0, "offset must be a multipe of 4");
    philox_offset_per_thread_ = offset;
}


c10::intrusive_ptr<c10::TensorImpl> TPUGeneratorImpl::get_state() const {
    static const size_t seed_size = sizeof(uint64_t);
    static const size_t offset_size = sizeof(int64_t);
    static const size_t total_size = seed_size + offset_size;
    auto state_tensor = at::detail::empty_cpu(
                            {(int64_t)total_size}, at::ScalarType::Byte,
                            c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
    auto rng_state = state_tensor.data_ptr<uint8_t>();
    auto current_seed = this->current_seed();
    auto offset = static_cast<int64_t>(this->philox_offset_per_thread());
    memcpy(rng_state, &current_seed, seed_size);
    memcpy(rng_state + seed_size, &offset, offset_size);
    return state_tensor.getIntrusivePtr();
}

void TPUGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
    static const size_t seed_size = sizeof(uint64_t);
    static const size_t offset_size = sizeof(int64_t);
    static const size_t total_size = seed_size + offset_size;

    at::detail::check_rng_state(new_state);

    bool no_philox_seed = false;
    auto new_state_size = new_state.numel();
    if (new_state_size == total_size - offset_size) {
        no_philox_seed = true;
    } else {
        TORCH_CHECK(new_state_size == total_size, "RNG state is wrong size");
    }

    uint64_t input_seed;
    auto new_rng_state = new_state.data<uint8_t>();
    memcpy(&input_seed, new_rng_state, seed_size);
    this->set_current_seed(input_seed);
    int64_t philox_offset = 0;
    if (!no_philox_seed) {
        memcpy(&philox_offset, new_rng_state + seed_size, offset_size);
    }
    this->set_philox_offset_per_thread(static_cast<uint64_t>(philox_offset));
}

c10::DeviceType TPUGeneratorImpl::device_type() {
    return c10::DeviceType::PrivateUse1;
}

std::shared_ptr<TPUGeneratorImpl> TPUGeneratorImpl::clone() const {
    return std::shared_ptr<TPUGeneratorImpl>(this->clone_impl());
}

TPUGeneratorImpl* TPUGeneratorImpl::clone_impl() const {
    auto gen = new TPUGeneratorImpl(this->device().index());
    gen->set_current_seed(this->seed_);
    gen->set_philox_offset_per_thread(this->philox_offset_per_thread_);
    return gen;
}

};