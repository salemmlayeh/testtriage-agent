# High failure rate (39%) driven by persistent communication errors on CAN bus

Severity: high

Run 3 experienced 117 failures out of 300 tests, with 89 identified as communication-related.
WindowLifter and Wiper ECUs show the highest failure rates at 0.48 and 0.44 respectively.
The high volume of persistent timeouts suggests a shared communication bus bottleneck or physical layer degradation.
Next steps include inspecting CAN bus wiring/termination and reviewing communication stack configurations for bandwidth issues.
