#include "sandboxNeg_FN/TelemetryCollector.hpp"
#include <iostream>
#include <cstdlib>

namespace sandboxNeg_FN {

TelemetryCollector::TelemetryCollector() {}

void TelemetryCollector::startCollection() {
    std::cout << "[sandboxNeg_FN::TelemetryCollector] Starting telemetry." << std::endl;
    system("python3 /opt/telemetrycollector.py &");
}

void TelemetryCollector::stopCollection() {
    std::cout << "[sandboxNeg_FN::TelemetryCollector] Stopping telemetry." << std::endl;
    system("pkill -f telemetrycollector.py");
}

} // namespace sandbox2
