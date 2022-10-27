#ifndef CONFIG_H_
#define CONFIG_H_

#define NPU_SHIFT               (CONFIG_NPU_SHIFT)
#define EU_SHIFT                (CONFIG_EU_SHIFT)
#ifndef LOCAL_MEM_SIZE
#define LOCAL_MEM_SIZE          (1<<CONFIG_LOCAL_MEM_ADDRWIDTH)
#endif
#ifndef L2_SRAM_SIZE
#define L2_SRAM_SIZE            (CONFIG_L2_SRAM_SIZE)
#endif
#ifndef LOCAL_MEM_BANKS
#define LOCAL_MEM_BANKS         (CONFIG_LOCAL_MEM_BANKS)
#endif
#define STATIC_MEM_SIZE         (CONFIG_STATIC_MEM_SIZE)

#ifndef NPU_NUM
#define NPU_NUM                 (1<<NPU_SHIFT)
#endif
#define NPU_MASK                (NPU_NUM - 1)
#define EU_NUM                  (1<<EU_SHIFT)
#define EU_NUM_32BIT            (EU_NUM)
#define EU_NUM_16BIT            (EU_NUM_32BIT << 1)
#define EU_NUM_8BIT             (EU_NUM_16BIT << 1)

#define MAX_ROI_NUM             200
#define KERNEL_MEM_SIZE         0
#define ALIGN_BYTES             (EU_NUM * sizeof(float))
#define LOCAL_BANK_SIZE         (LOCAL_MEM_SIZE / LOCAL_MEM_BANKS)


#endif
