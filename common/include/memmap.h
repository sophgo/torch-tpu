#ifndef MEMMAP_H
#define MEMMAP_H

#include "config.h"

// =============================================
// The following is allocation for static memory
// For lookup table
#define SFU_TABLE_SIZE              256
#define SFU_TAYLOR_TABLE_SIZE       32
#define SFU_TAYLOR_L_TABLE_SIZE     64
#define ERF_TAYLOR_SIZE             16
#define STATIC_MEM_OFFSET           0
#define SERIAL_NUMBER_SIZE          64
#define SIN_TAYLOR_SIZE             32
#define COS_TAYLOR_SIZE             32
#define ARCSIN_TAYLOR_SIZE          64
#define TAN_TAYLOR_SIZE             32
#define EXP_TABLE_OFFSET            (STATIC_MEM_OFFSET)
#define EXP_TAYLOR_OFFSET           (EXP_TABLE_OFFSET + SFU_TABLE_SIZE * sizeof(float))
#define LOG_TAYLOR_OFFSET           (EXP_TAYLOR_OFFSET + SFU_TAYLOR_TABLE_SIZE * sizeof(float))
#define ERF_TAYLOR_OFFSET           (LOG_TAYLOR_OFFSET + SFU_TAYLOR_L_TABLE_SIZE * sizeof(float))
#define SERIAL_NUMBER_OFFSET        (ERF_TAYLOR_OFFSET + ERF_TAYLOR_SIZE * sizeof(float))
#define SIN_TAYLOR_OFFSET           (SERIAL_NUMBER_OFFSET + SERIAL_NUMBER_SIZE * sizeof(float))
#define COS_TAYLOR_OFFSET           (SIN_TAYLOR_OFFSET + SIN_TAYLOR_SIZE * sizeof(float))
#define ARCSIN_TAYLOR_OFFSET        (COS_TAYLOR_OFFSET + COS_TAYLOR_SIZE * sizeof(float))
#define TAN_TAYLOR_OFFSET           (ARCSIN_TAYLOR_OFFSET + ARCSIN_TAYLOR_SIZE * sizeof(float))
#define SMEM_STATIC_END_OFFSET      (TAN_TAYLOR_OFFSET + TAN_TAYLOR_SIZE * sizeof(float))
// ============================================
// SMEM_STATIC_END_OFFSET must <= STATIC_MEM_SHARE_OFFSET

#define LOCAL_MEM_START_ADDR           0x08000000UL
#define LOCAL_MEM_ADDRWIDTH            CONFIG_LOCAL_MEM_ADDRWIDTH

#define STATIC_MEM_START_ADDR          0x09000000UL
#define STATIC_MEM_SHARE_SIZE          (0x4000)          // 16KB for share memory.
#define STATIC_MEM_SHARE_OFFSET        (STATIC_MEM_SIZE - STATIC_MEM_SHARE_SIZE)
#define SHARE_MEM_START_ADDR           (STATIC_MEM_START_ADDR + STATIC_MEM_SHARE_OFFSET)
// msg share mem total 4K words, 2k words for each channel
#define SHAREMEM_SIZE_BIT              11
#define SHAREMEM_MASK                  ((1 << SHAREMEM_SIZE_BIT) - 1)

#define DEBUG_START_ADDR               0x09008000UL

#define L2_SRAM_START_ADDR             0x10000000UL

#define GLOBAL_MEM_START_ADDR_ARM      0x100000000UL
#define GLOBAL_MEM_CMD_START_OFFSET    0x0
#define GLOBAL_MEM_START_ADDR          0x100000000UL

#define BD_REG_COUNT                   (32)
#define BD_ENGINE_COMMAND_NUM          (BD_REG_COUNT)

#define GDMA_REG_COUNT                 (24)
#define GDMA_ENGINE_COMMAND_NUM        GDMA_REG_COUNT

#define GDE_REG_COUNT                  (40)
#define GDE_ENGINE_COMMAND_NUM         GDE_REG_COUNT

#define SORT_REG_COUNT                 (36)

#define NMS_REG_COUNT                  (33)

#define SPI_BASE_ADDR                  0x06000000UL
#define TOP_REG_CTRL_BASE_ADDR         (0x50010000UL)
#define SHARE_REG_BASE_ADDR            (TOP_REG_CTRL_BASE_ADDR + 0x80)
#define SHARE_REG_MESSAGE_WP           0
#define SHARE_REG_MESSAGE_RP           1
#define SHARE_REG_MSI_DATA             3
#define SHARE_REG_A53LITE_READ_REG_ADDR   8
#define SHARE_REG_FW_STATUS            9
#define SHARE_REG_A53LITE_FW_MODE      13
#define SHARE_REG_A53LITE_FW_LOG_RP    11
#define SHARE_REG_A53LITE_FW_LOG_WP    12
#define SHARE_REG_CNT                  14

#define I2C0_REG_CTRL_BASE_ADDR        0x5001a000UL
#define I2C1_REG_CTRL_BASE_ADDR        0x5001c000UL
#define I2C2_REG_CTRL_BASE_ADDR        0x5001e000UL

#define NV_TIMER_CTRL_BASE_ADDR        0x50010180UL
#define OS_TIMER_CTRL_BASE_ADDR        0x50022000UL
#define INT0_CTRL_BASE_ADDR            0x50023000UL
#define EFUSE_BASE                     0x50028000UL
#define GPIO_CTRL_BASE_ADDR            0x50027000UL
#define PCIE_BUS_SCAN_ADDR             0x60000000UL
#define PWM_CTRL_BASE_ADDR             0x50029000UL
#define DDR_CTRL_BASE_ADDR             0x68000000UL
#define GPIO0_CTRL_BASE_ADDR           0x50027000UL

#define UART_CTRL_BASE_ADDR            0x50118000UL
// There're 8 device lock register, the lower 4 for PCIe and upper 4 for SoC.
#define TOP_REG_DEVICE_LOCK0           0x50010040UL
#define TOP_REG_DEVICE_LOCK1           0x50010044UL
#define TOP_REG_DEVICE_LOCK2           0x50010048UL
#define TOP_REG_DEVICE_LOCK3           0x5001004cUL
#define TOP_GP_REG_ARM9_IRQ_STATUS     0x500100b8UL  // TODO for A53 Lite
#define TOP_GP_REG_ARM9_IRQ_SET        0x50010190UL  // TODO for A53 Lite
#define TOP_GP_REG_ARM9_IRQ_CLR        0x50010194UL  // TODO for A53 Lite
#define TOP_GP_REG_A53_IRQ_STATUS      0x500100bcUL
#define TOP_GP_REG_A53_IRQ_SET         0x50010198UL
#define TOP_GP_REG_A53_IRQ_CLR         0x5001019cUL
#define TOP_REG_DEVICE_LOCK_CDMA       TOP_REG_DEVICE_LOCK0
#define TOP_REG_DEVICE_LOCK_CLK        TOP_REG_DEVICE_LOCK1
#define TOP_REG_CLK_ENABLE0            0x50010800UL
#define TOP_REG_CLK_ENABLE1            0x50010804UL

#define GDMA_ENGINE_BASE_ADDR_AHB      0x58000000UL
#define BD_ENGINE_BASE_ADDR_AHB        0x58001000UL
#define MMU_ENGINE_BASE_ADDR           0x58002000UL
#define CDMA_ENGINE_BASE_ADDR          0x58003000UL
#define GDE_BASE_ADDR                  0x2008000UL
#define SORT_BASE_ADDR                 0x2009000UL
#define NMS_BASE_ADDR                  0x200a000UL

#define GDMA_ENGINE_BASE_ADDR          0x58000000UL
#define BD_ENGINE_BASE_ADDR            0x58001000UL

#define BD_ENGINE_MAIN_CTRL            (BD_ENGINE_BASE_ADDR + 0x00000100)
#define BD_ENGINE_MAIN_CTRL_AHB        (BD_ENGINE_BASE_ADDR_AHB + 0x00000100)
#define BDC_CMD_BASE_ADDR              (BD_ENGINE_BASE_ADDR + 0x00000000)
#define BDC_CMD_BASE_ADDR_AHB          (BD_ENGINE_BASE_ADDR_AHB + 0x00000000) // for arm read
#define BDC_INT_CLR                    (1)

#define GDMA_ENGINE_MAIN_CTRL          (GDMA_ENGINE_BASE_ADDR + 0x100)
#define GDMA_ENGINE_MAIN_CTRL_AHB      (GDMA_ENGINE_BASE_ADDR_AHB + 0x100)
#define GDMA_CMD_BASE_ADDR             (GDMA_ENGINE_BASE_ADDR + 0x0)
#define GDMA_CMD_BASE_ADDR_AHB         (GDMA_ENGINE_BASE_ADDR_AHB + 0x0) // for arm read

#define GDE_CMD_BASE_ADDR              (GDE_BASE_ADDR)
#define SORT_CMD_BASE_ADDR             (SORT_BASE_ADDR + 0x100)
#define NMS_CMD_BASE_ADDR              (NMS_BASE_ADDR + 0x100)


#define COUNT_RESERVED_DDR_INSTR         0x1000000
//#define COUNT_RESERVED_DDR_SWAP        0x1000000
//#define COUNT_RESERVED_DDR_IMAGE_SCALE 0x2000000

#define API_MESSAGE_EMPTY_SLOT_NUM 2


#endif  /* MEMMAP_H */
