#include "sandboxPos_FP/TelemetryCollector.hpp"
#include <cstdlib>
#include <iostream>

namespace sandboxPos_FP {

TelemetryCollector::TelemetryCollector() {}

void TelemetryCollector::startCollection() {
    std::cout << "[TelemetryCollector] Starting telemetry collection..." << std::endl;
    // Launch eBPF or alternative telemetry collection asynchronously here
    system("python3 /opt/telemetrycollector.py &");
}

void TelemetryCollector::stopCollection() {
    std::cout << "[TelemetryCollector] Stopping telemetry collection..." << std::endl;
    system("pkill -f telemetrycollector.py");
}

} // namespace sandbox1