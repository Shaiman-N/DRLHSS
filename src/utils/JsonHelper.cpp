#include "utils/JsonHelper.hpp"
#include <iostream>
#include <fstream>
#include <nlohmann/json-schema.hpp>

using nlohmann::json;
using nlohmann::json_schema::json_validator;

json JsonHelper::parseString(const std::string& jsonStr) {
    try {
        return json::parse(jsonStr);
    } catch (const json::parse_error& ex) {
        std::cerr << "[JsonHelper] JSON parse error: " << ex.what() << std::endl;
        throw;
    }
}

std::string JsonHelper::serialize(const json& j) {
    return j.dump();
}

bool JsonHelper::validateSchema(const json& jsonData, const json& schema) {
    try {
        // Use the default validator constructor (no custom loader).
        json_validator validator;
        validator.set_root_schema(schema);
        validator.validate(jsonData);
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "[JsonHelper] JSON schema validation error: " << ex.what() << std::endl;
        return false;
    }
}