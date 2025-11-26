#pragma once
#include <string>
#include "json.hpp"

class JsonHelper
{
    public : 
        static nlohmann::json parseString(const std::string& jsonStr);
        static std::string serialize(const nlohmann::json& j);
        static bool validateSchema(const nlohmann::json& jsonData, const nlohmann::json& schema);
};