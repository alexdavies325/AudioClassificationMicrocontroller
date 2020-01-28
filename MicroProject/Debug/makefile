################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include X-CUBE-AI/App/subdir.mk
-include Drivers/STM32F7xx_HAL_Driver/Src/subdir.mk
-include Drivers/CMSIS/DSP/Source/SupportFunctions/subdir.mk
-include Drivers/CMSIS/DSP/Source/MatrixFunctions/subdir.mk
-include Drivers/CMSIS/DSP/Source/BasicMathFunctions/subdir.mk
-include Core/Startup/subdir.mk
-include Core/Src/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 
EXECUTABLES += \
FinalAudioClassificationPPGC8.elf \

SIZE_OUTPUT += \
default.size.stdout \

OBJDUMP_LIST += \
FinalAudioClassificationPPGC8.list \

OBJCOPY_BIN += \
FinalAudioClassificationPPGC8.bin \


# All Target
all: FinalAudioClassificationPPGC8.elf secondary-outputs

# Tool invocations
FinalAudioClassificationPPGC8.elf: $(OBJS) $(USER_OBJS) /home/alex/Alex/STM32CubeIDE/workspace_1.1.0/FinalAudioClassificationPPGC8/STM32F746ZGTX_FLASH.ld
	arm-none-eabi-gcc -o "FinalAudioClassificationPPGC8.elf" @"objects.list" $(USER_OBJS) $(LIBS) -mcpu=cortex-m7 -T"/home/alex/Alex/STM32CubeIDE/workspace_1.1.0/FinalAudioClassificationPPGC8/STM32F746ZGTX_FLASH.ld" --specs=nosys.specs -Wl,-Map="FinalAudioClassificationPPGC8.map" -Wl,--gc-sections -static -L../Middlewares/ST/AI/Lib --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -u _printf_float -Wl,--start-group -lc -lm -Wl,--end-group
	@echo 'Finished building target: $@'
	@echo ' '

default.size.stdout: $(EXECUTABLES)
	arm-none-eabi-size  $(EXECUTABLES)
	@echo 'Finished building: $@'
	@echo ' '

FinalAudioClassificationPPGC8.list: $(EXECUTABLES)
	arm-none-eabi-objdump -h -S $(EXECUTABLES) > "FinalAudioClassificationPPGC8.list"
	@echo 'Finished building: $@'
	@echo ' '

FinalAudioClassificationPPGC8.bin: $(EXECUTABLES)
	arm-none-eabi-objcopy  -O binary $(EXECUTABLES) "FinalAudioClassificationPPGC8.bin"
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) *
	-@echo ' '

secondary-outputs: $(SIZE_OUTPUT) $(OBJDUMP_LIST) $(OBJCOPY_BIN)

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
