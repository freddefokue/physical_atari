# Building a Digital I/O Controller for Atari 2600+

This document walks you through building a **digital I/O controller module** that connects to the **Atari 2600 joystick port** and simulates joystick inputs (up/down/left/right/fire). It uses a digital I/O device such as the **MCC USB-1024LS**, wired to a **DB9 Atari joystick cable** to send digital signals directly to the Atari console.

The controller works by pulling specific pins on the DB9 joystick port **low (to ground)**, mimicking how the real joystick functions. This setup is useful for experimenting with Atari hardware from modern control systems.

---

## Hardware Overview

### Digital I/O Devices

You will need a USB digital I/O module with TTL-level (5V) outputs. Recommended options:

| Device | Description | Notes |
|--------|-------------|-------|
| **[MCC USB-1024LS](https://microdaq.com/usb-1024ls-24-bit-digital-input-output-i-o-module.php)** | 24 digital I/O lines, 5V TTL | Well-documented and supported |
| **Arduino Nano + Screw Terminal Shield** | 14 digital I/O lines | Requires custom sketch |

---

## Atari Joystick Port Pinout (DB9)

Refer to the [Atari joystick port pinout](https://en.wikipedia.org/wiki/Atari_joystick_port), which shows the connector **as seen from the front (on the Atari console)**:

| Pin | Function     | Notes |
|-----|--------------|-------|
| 1   | Up           | Active LOW |
| 2   | Down         | Active LOW |
| 3   | Left         | Active LOW |
| 4   | Right        | Active LOW |
| 5   | Paddle B     | **Not used** |
| 6   | Fire Button  | Active LOW |
| 7   | +5V Power    | **Unused** |
| 8   | Ground (GND) | Shared return path |
| 9   | Paddle A     | **Not used** |

> If you're using a **male DB9-to-bare wire cable**, the pinout may appear mirrored. Use continuity testing to confirm pin mapping.

---

## Identifying Wires via Continuity Testing

To build the controller, you'll need to match each wire in the DB9 cable to its pin. Wire colors are often non-standard.

### Steps:

1. Insert **paper clips into the front of the DB9 connector** to make contact with the pins.
2. Set your **multimeter to continuity mode** (beep or ohm check).
3. Touch one multimeter probe to a paper clip in **Pin 1 (Up)**.
4. Touch the other probe to each wire end until you hear a beep.
5. Repeat for Pins 2–4, 6, and 8 (GND).

Write down the wire color associated with each pin for reference during wiring.

> Avoid touching adjacent paper clips at the same time — you may short pins during testing.

---

## Wiring the Controller Module

Once you have mapped the wires, connect them to the I/O device.

### Signal Logic

- The Atari expects **active LOW signals**: pulling the signal to **GND = pressed**.
- Set digital output pins **LOW (0V)** to press, **HIGH (5V)** to release.

### MCC USB-1024LS: Port A Wiring

If using the MCC USB-1024LS, connect the joystick wires to **Port A**, using pins **24–28 for control**, and **29 for GND**:

| Function | DB9 Pin | DAQ Port | Terminal Pin # | Bit | Notes |
|----------|---------|----------|----------------|------|-------|
| Up       | 1       | Port A   | 24             | P0.0 | Press = LOW |
| Down     | 2       | Port A   | 25             | P0.1 | Press = LOW |
| Left     | 3       | Port A   | 26             | P0.2 | Press = LOW |
| Right    | 4       | Port A   | 27             | P0.3 | Press = LOW |
| Fire     | 6       | Port A   | 28             | P0.4 | Press = LOW |
| GND      | 8       | GND      | 29             | –    | Required for signal return |

**Important**: Connect the GND wire (pin 8) from the joystick cable to **terminal pin 29 (GND)** on the DAQ to complete the circuit.

---

## Cable Prep & Strain Relief

When handling joystick cables:

1. **Strip each wire** ~5mm from the end. Be careful with cables that have **nylon/tinsel** insulation inside - trim or **singe** with a lighter.
2. Insert stripped wires into **screw terminals** on the I/O module.
3. Double-check wire mapping before applying power.

### Strain Relief

Use a **cable tie and cable tie gun** to secure the joystick cable to the screw terminal block. This prevents stress on the connections and improves durability.

---
