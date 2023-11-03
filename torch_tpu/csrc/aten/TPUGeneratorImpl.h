#pragma once

#include <c10/core/GeneratorImpl.h>
#include <c10/core/TensorImpl.h>

namespace at_tpu {

struct PhiloxTpuState {
  PhiloxTpuState() = default;
  PhiloxTpuState(const PhiloxTpuState&) = default;
  // Called if graph capture is not underway
  PhiloxTpuState(uint64_t seed,
                  uint64_t offset) {
    seed_ = seed;
    offset_.val = offset;
  }
  // Called if graph capture is underway
  PhiloxTpuState(uint64_t seed,
                  int64_t* offset_extragraph,
                  uint32_t offset_intragraph) {
    seed_ = seed;
    offset_.ptr = offset_extragraph;
    offset_intragraph_ = offset_intragraph;
    captured_ = true;
  }


  union Payload {
    uint64_t val;
    int64_t* ptr;
  };

  uint64_t seed_;
  Payload offset_;
  uint32_t offset_intragraph_{0};
  bool captured_ = false;
};


struct TPUGeneratorImpl : public c10::GeneratorImpl {
  TPUGeneratorImpl(c10::DeviceIndex device_index = -1);
  ~TPUGeneratorImpl() = default;

  // NPUGeneratorImpl methods
  std::shared_ptr<TPUGeneratorImpl> clone() const;
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  void set_state(const c10::TensorImpl& new_state) override;
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;
  void set_philox_offset_per_thread(uint64_t offset);
  uint64_t philox_offset_per_thread() const;
  void capture_prologue(int64_t* offset_extragraph);
  uint64_t capture_epilogue();
  PhiloxTpuState philox_npu_state(uint64_t increment);

  // Temporarily accommodates call sites that use philox_engine_inputs.
  // Allows incremental refactor of call sites to use philox_npu_state.
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);
  static c10::DeviceType device_type();

private:
  TPUGeneratorImpl* clone_impl() const override;
  uint64_t seed_ = c10::default_rng_seed_val;
  uint64_t philox_offset_per_thread_ = 0;
  int64_t* offset_extragraph_ = nullptr;
  uint32_t offset_intragraph_ = 0;
  bool graph_expects_this_gen_ = false;

};

}; // namespace aten_tpu