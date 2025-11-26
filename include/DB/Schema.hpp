#pragma once
#include <string>
#include <nlohmann/json.hpp>

class DBSchema
{
    public :
     static nlohmann::json getTelemetrySchema();
     static nlohmann::json getExperienceSchema();
     static bool validateTelemetry(const nlohmann::json& telemetry);
     static bool validateExperience(const nlohmann::json& experience);
};