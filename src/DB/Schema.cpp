#include "DB/Schema.hpp"
#include "utils/JsonHelper.hpp"

using nlohmann::json;

using nlohmann::json;

json DBSchema::getTelemetrySchema()
{
    return json::parse(R"({
        "type":"objecy",
        "properties" : {
        "fo":{"type" : "number"},
        "f1":{"type" : "number"},
        "f2":{"type" : "number"},
        },
        "required": ["f0", "f1", f2"],
        "additionalProperties" : true
    })");
}

json DBSchema::getExperienceSchema()
{
    return json::parse(R"({
            "type" : "object",
            "properties" : 
            {
                "state" : {"type" : "array", "items" : {"type":"number"}},
                "action" : {"type" : "integer"},
                "reward" {"type" : "number"},
                "next_state" : {"type" : "array", "items" : {"type":"number"}},
                "done" : {"type" : "boolean"},
            },
            "required" : ["state", "action", "reward", "next_state", "done"],
            "additionalProperties":false
        })");
}

bool DBSchema::validateTelemetry(const json& telemetry)
{
    return JsonHelper::validateSchema(telemetry, getTelemetrySchema());
}

bool DBSchema::validateExperience(const json& experience)
{
    return JsonHelper::validateSchema(experience, getExperienceSchema());
}