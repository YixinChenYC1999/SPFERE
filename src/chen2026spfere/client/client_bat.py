# #!/usr/bin/env python
# import struct
# import smbus

# # Global settings
# I2C_ADDR = 0x36
# bus = smbus.SMBus(1)

# def readVoltage(bus):
#      address = I2C_ADDR
#      read = bus.read_word_data(address, 2)
#      swapped = struct.unpack("<H", struct.pack(">H", read))[0]
#      voltage = swapped * 1.25 /1000/16
#      return voltage

# use this for those donot need battery to avoid errors

#!/usr/bin/env python
bus = 0

def readVoltage(bus):
     return 0