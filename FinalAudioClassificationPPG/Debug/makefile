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
FinalAudioClassificationPPG.elf \

SIZE_OUTPUT += \
default.size.stdout \

OBJDUMP_LIST += \
FinalAudioClassificationPPG.list \

OBJCOPY_BIN += \
FinalAudioClassificationPPG.bin \


# All Target
all: FinalAudioClassificationPPG.elf secondary-outputs

# Tool invocations
FinalAudioClassificationPPG.elf: $(OBJS) $(USER_OBJS) /home/alex/Alex/STM32CubeIDE/workspace_1.1.0/FinalAudioClassificationPPG/STM32F746ZGTX_FLASH.ld
	arm-none-eabi-gcc -o "FinalAudioClassificationPPG.elf" @"objects.list" $(USER_OBJS) $(LIBS) -mcpu=cortex-m7 -T"/home/alex/Alex/STM32CubeIDE/workspace_1.1.0/FinalAudioClassificationPPG/STM32F746ZGTX_FLASH.ld" --specs=nosys.specs -Wl,-Map="FinalAudioClassificationPPG.map" -Wl,--gc-sections -static -L../Middlewares/ST/AI/Lib --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -u _printf_float -Wl,--start-group -lc -lm -Wl,--end-group
	@echo 'Finished building target: $@'
	@echo ' '

default.size.stdout: $(EXECUTABLES)
	arm-none-eabi-size  $(EXECUTABLES)
	@echo 'Finished building: $@'
	@echo ' '

FinalAudioClassificationPPG.list: $(EXECUTABLES)
	arm-none-eabi-objdump -h -S $(EXECUTABLES) > "FinalAudioClassificationPPG.list"
	@echo 'Finished building: $@'
	@echo ' '

FinalAudioClassificationPPG.bin: $(EXECUTABLES)
	arm-none-eabi-objcopy  -O binary $(EXECUTABLES) "FinalAudioClassificationPPG.bin"
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
