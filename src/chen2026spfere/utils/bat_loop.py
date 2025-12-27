from chen2026spfere.client.client_bat import readVoltage, bus
import time

while True:
    print("******************")
    print("Voltage:%5.2fV" % readVoltage(bus))
    time.sleep(1)