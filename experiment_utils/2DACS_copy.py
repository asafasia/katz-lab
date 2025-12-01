import serial
import time


# Open serial connection to Arduino (adjust COM port if needed)

arduino = serial.Serial('COM7', 115200, timeout=1)
time.sleep(2)  # Wait for Arduino to initialize

# Function to set voltage on a specific channel
def set_voltage(channel, voltage):
    command = f"SET,{channel},{voltage}\r"
    arduino.write(command.encode())
    # time.sleep(0.1)  # Give Arduino time to respond

    # Read and print response from Arduino
    while arduino.in_waiting:
        response = arduino.readline().decode(errors='ignore').strip()
        if response:
            print("Arduino responded:", response)

# Initially set both channels (2 and 4) to 6V
set_voltage(5, 7)
set_voltage(7, 7)
time.sleep(2)
# After a small delay, change the voltage on channel 2 to 3V, leaving channel 4 unchanged
# time.sleep(2)
# set_voltage(2, 3)







# import serial
# import time
#
# # Open serial connection to Arduino (replace with correct COM port)
# arduino = serial.Serial('COM10', 115200, timeout=1)
# time.sleep(2)  # Wait for Arduino to initialize
#
# # Function to set voltage on a specific channel
# def set_voltage(channel, voltage):
#     command = f"SET,{channel},{voltage}\r"
#     arduino.write(command.encode())
#     time.sleep(1)  # Give it time to process the command
#
# # Initially set both channels (2 and 4) to 6V
# set_voltage(2, 6)
# set_voltage(4, 6)
#
# # After a small delay, change the voltage on channel 2 to 3V, leaving channel 4 unchanged
# time.sleep(2)  # Wait for a moment to make sure initial voltages are set
# set_voltage(2, 3)  # Change channel 2 to 3V, channel 4 stays at 6V
#
# # Done!
