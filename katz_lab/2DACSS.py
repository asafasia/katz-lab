import serial
import time

# Open the serial connection to Arduino (adjust COM port if needed)
arduino = serial.Serial('COM10', 115200, timeout=1)
time.sleep(2)  # Wait for Arduino to initialize

print("Connected to Arduino.")
print("Type commands like SET,2,3 or 'exit' to quit.\n")

try:
    while True:
        command = input("Enter command: ")
        if command.lower() == 'exit':
            break
        if command.strip() == '':
            continue

        # Send command to Arduino with carriage return as terminator
        arduino.write((command + '\r').encode())

        # Wait for and print response(s)
        time.sleep(0.1)
        while arduino.in_waiting:
            response = arduino.readline().decode(errors='ignore').strip()
            if response:
                print("Arduino responded:", response)

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    arduino.close()
    print("Serial connection closed.")
