project(FIRMWARE C)

if (USING_CMODEL)
    include(cmodel.cmake)
elseif (PCIE_MODE)
    include(pcie.cmake)
elseif (SOC_MODE)
    include(soc.cmake)
endif()
