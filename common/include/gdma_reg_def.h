#ifndef __GDMA_REG_DEF_H__
#define __GDMA_REG_DEF_H__


// gdma reg id defines
#define GDMA_ID_INTR_EN                           {0, 1}
#define GDMA_ID_STRIDE_ENABLE                     {1, 1}
#define GDMA_ID_NCHW_COPY                         {2, 1}
#define GDMA_ID_CMD_SHORT                         {3, 1}
#define GDMA_ID_DECOMPRESS_ENABLE                 {4, 1}
#define GDMA_ID_CMD_ID_EN                         {5, 4}
#define GDMA_ID_CMD_ID                            {9, 20}
#define GDMA_ID_CMD_TYPE                          {32, 4}
#define GDMA_ID_CMD_SPECIAL_FUNCTION              {36, 3}
#define GDMA_ID_FILL_CONSTANT_EN                  {39, 1}
#define GDMA_ID_SRC_DATA_FORMAT                   {40, 3}
#define GDMA_ID_MASK_DATA_FORMAT                  {43, 3}  // Also for index
#define GDMA_ID_CMD_ID_DEP                        {64, 20}
#define GDMA_ID_CONSTANT_VALUE                    {96, 32}
#define GDMA_ID_SRC_NSTRIDE                       {128, 32}
#define GDMA_ID_SRC_CSTRIDE                       {160, 32}
#define GDMA_ID_SRC_HSTRIDE                       {192, 32}
#define GDMA_ID_SRC_WSTRIDE                       {224, 32}
#define GDMA_ID_DST_NSTRIDE                       {256, 32}
#define GDMA_ID_DST_CSTRIDE                       {288, 32}
#define GDMA_ID_DST_HSTRIDE                       {320, 32}
#define GDMA_ID_DST_WSTRIDE                       {352, 32}
#define GDMA_ID_SRC_NSIZE                         {384, 16}
#define GDMA_ID_SRC_CSIZE                         {400, 16}
#define GDMA_ID_SRC_HSIZE                         {416, 16}
#define GDMA_ID_SRC_WSIZE                         {432, 16}
#define GDMA_ID_DST_NSIZE                         {448, 16}
#define GDMA_ID_DST_CSIZE                         {464, 16}
#define GDMA_ID_DST_HSIZE                         {480, 16}
#define GDMA_ID_DST_WSIZE                         {496, 16}
#define GDMA_ID_SRC_START_ADDR_L32                {512, 32}
#define GDMA_ID_SRC_START_ADDR_H8                 {544, 8}
#define GDMA_ID_DST_START_ADDR_L32                {576, 32}
#define GDMA_ID_DST_START_ADDR_H8                 {608, 8}
#define GDMA_ID_MASK_START_ADDR_L32               {640, 32}  // Also for index
#define GDMA_ID_MASK_START_ADDR_H8                {672, 32}  // Also for index
#define GDMA_ID_LOCALMEM_MASK_L32                 {704, 32}
#define GDMA_ID_LOCALMEM_MASK_H32                 {736, 32}


// gdma pack defines
#define GDMA_PACK_INTR_EN(val)                    {GDMA_ID_INTR_EN, (val)}
#define GDMA_PACK_STRIDE_ENABLE(val)              {GDMA_ID_STRIDE_ENABLE, (val)}
#define GDMA_PACK_NCHW_COPY(val)                  {GDMA_ID_NCHW_COPY, (val)}
#define GDMA_PACK_CMD_SHORT(val)                  {GDMA_ID_CMD_SHORT, (val)}
#define GDMA_PACK_DECOMPRESS_ENABLE(val)          {GDMA_ID_DECOMPRESS_ENABLE, (val)}
#define GDMA_PACK_CMD_ID_EN(val)                  {GDMA_ID_CMD_ID_EN, (val)}
#define GDMA_PACK_CMD_ID(val)                     {GDMA_ID_CMD_ID, (val)}
#define GDMA_PACK_CMD_TYPE(val)                   {GDMA_ID_CMD_TYPE, (val)}
#define GDMA_PACK_CMD_SPECIAL_FUNCTION(val)       {GDMA_ID_CMD_SPECIAL_FUNCTION, (val)}
#define GDMA_PACK_FILL_CONSTANT_EN(val)           {GDMA_ID_FILL_CONSTANT_EN, (val)}
#define GDMA_PACK_SRC_DATA_FORMAT(val)            {GDMA_ID_SRC_DATA_FORMAT, (val)}
#define GDMA_PACK_MASK_DATA_FORMAT(val)           {GDMA_ID_MASK_DATA_FORMAT, (val)}
#define GDMA_PACK_CMD_ID_DEP(val)                 {GDMA_ID_CMD_ID_DEP, (val)}
#define GDMA_PACK_CONSTANT_VALUE(val)             {GDMA_ID_CONSTANT_VALUE, (val)}
#define GDMA_PACK_SRC_NSTRIDE(val)                {GDMA_ID_SRC_NSTRIDE, (val)}
#define GDMA_PACK_SRC_CSTRIDE(val)                {GDMA_ID_SRC_CSTRIDE, (val)}
#define GDMA_PACK_SRC_HSTRIDE(val)                {GDMA_ID_SRC_HSTRIDE, (val)}
#define GDMA_PACK_SRC_WSTRIDE(val)                {GDMA_ID_SRC_WSTRIDE, (val)}
#define GDMA_PACK_DST_NSTRIDE(val)                {GDMA_ID_DST_NSTRIDE, (val)}
#define GDMA_PACK_DST_CSTRIDE(val)                {GDMA_ID_DST_CSTRIDE, (val)}
#define GDMA_PACK_DST_HSTRIDE(val)                {GDMA_ID_DST_HSTRIDE, (val)}
#define GDMA_PACK_DST_WSTRIDE(val)                {GDMA_ID_DST_WSTRIDE, (val)}
#define GDMA_PACK_SRC_NSIZE(val)                  {GDMA_ID_SRC_NSIZE, (val)}
#define GDMA_PACK_SRC_CSIZE(val)                  {GDMA_ID_SRC_CSIZE, (val)}
#define GDMA_PACK_SRC_HSIZE(val)                  {GDMA_ID_SRC_HSIZE, (val)}
#define GDMA_PACK_SRC_WSIZE(val)                  {GDMA_ID_SRC_WSIZE, (val)}
#define GDMA_PACK_DST_NSIZE(val)                  {GDMA_ID_DST_NSIZE, (val)}
#define GDMA_PACK_DST_CSIZE(val)                  {GDMA_ID_DST_CSIZE, (val)}
#define GDMA_PACK_DST_HSIZE(val)                  {GDMA_ID_DST_HSIZE, (val)}
#define GDMA_PACK_DST_WSIZE(val)                  {GDMA_ID_DST_WSIZE, (val)}
#define GDMA_PACK_SRC_START_ADDR_L32(val)         {GDMA_ID_SRC_START_ADDR_L32, (val)}
#define GDMA_PACK_SRC_START_ADDR_H8(val)          {GDMA_ID_SRC_START_ADDR_H8, (val)}
#define GDMA_PACK_DST_START_ADDR_L32(val)         {GDMA_ID_DST_START_ADDR_L32, (val)}
#define GDMA_PACK_DST_START_ADDR_H8(val)          {GDMA_ID_DST_START_ADDR_H8, (val)}
#define GDMA_PACK_MASK_START_ADDR_L32(val)        {GDMA_ID_MASK_START_ADDR_L32, (val)}
#define GDMA_PACK_MASK_START_ADDR_H8(val)         {GDMA_ID_MASK_START_ADDR_H8, (val)}
#define GDMA_PACK_LOCALMEM_MASK_L32(val)          {GDMA_ID_LOCALMEM_MASK_L32, (val)}
#define GDMA_PACK_LOCALMEM_MASK_H32(val)          {GDMA_ID_LOCALMEM_MASK_H32, (val)}


#endif  // __GDMA_REG_DEF_H__
