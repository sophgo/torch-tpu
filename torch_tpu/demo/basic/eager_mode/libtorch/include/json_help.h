#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <optional>
#include <functional>
#include <cstring>

#include "picojson.h"

#define PicoJsonParse(val, jsonStr) \
    { \
        std::string err = picojson::parse(val, jsonStr); \
        if (!err.empty()) { \
            std::cerr << "JSON parsing error: " << err << std::endl; \
            exit(-1); \
        } \
    }

typedef struct {
    std::vector<std::string> input_params;
    std::vector<std::string> input_buffers;
    std::vector<std::string> user_outputs;
    std::vector<std::string> user_inputs;
    std::vector<int> output_buffers;// for finetune model (you will freeze some layers) , the output buffer num will change
    std::vector<int> output_params;// same with output_buffers
} JitModelInfo_t;

static const picojson::array emptyArray;

class JsonBuilder {
private:
    picojson::object obj;
    std::vector<picojson::array*> currentArray;  // 用于处理嵌套数组

public:
    // 添加字符串值
    JsonBuilder& addString(const std::string& key, const std::string& value) {
        if (currentArray.empty()) {
            obj[key] = picojson::value(value);
        } else {
            currentArray.back()->push_back(picojson::value(value));
        }
        return *this;
    }

    // 添加数值
    JsonBuilder& addNumber(const std::string& key, double value) {
        if (currentArray.empty()) {
            obj[key] = picojson::value(value);
        } else {
            currentArray.back()->push_back(picojson::value(value));
        }
        return *this;
    }

    // 添加布尔值
    JsonBuilder& addBool(const std::string& key, bool value) {
        if (currentArray.empty()) {
            obj[key] = picojson::value(value);
        } else {
            currentArray.back()->push_back(picojson::value(value));
        }
        return *this;
    }

    // 添加嵌套对象
    JsonBuilder& addObject(const std::string& key, const std::function<void(JsonBuilder&)>& initializer) {
        JsonBuilder subBuilder;
        initializer(subBuilder);  // 使用提供的初始化器来填充子对象
        obj[key] = subBuilder.build();
        return *this;
    }

    template<typename T>
    JsonBuilder& addElement(const T& value) {
        if (!currentArray.empty()) {
            // if constexpr (std::is_integral_v<T>){
            //     currentArray.back()->push_back(picojson::value(static_cast<double>(value)));
            // }else{
                currentArray.back()->push_back(picojson::value(value));
            // }

        }
        return *this;
    }

    JsonBuilder& beginArray(const std::string& key) {
        picojson::array arr;
        obj[key] = picojson::value(arr);
        currentArray.push_back(&obj[key].get<picojson::array>());
        return *this;
    }

    JsonBuilder& endArray() {
        if (!currentArray.empty()) {
            currentArray.pop_back();
        }
        return *this;
    }

    picojson::value build() {
        return picojson::value(obj);
    }
};

class JsonAccessor {
private:
    const picojson::value* value;

public:
    JsonAccessor() = default;
    explicit JsonAccessor(const picojson::value& val) : value(&val) {}

    std::optional<std::string> getString() const {
        if (value->is<std::string>()) {
            return value->get<std::string>();
        }
        return std::nullopt;
    }

    std::optional<double> getNumber() const {
        if (value->is<double>()) {
            return value->get<double>();
        }
        return std::nullopt;
    }

    std::optional<bool> getBool() const {
        if (value->is<bool>()) {
            return value->get<bool>();
        }
        return std::nullopt;
    }

    JsonAccessor operator[](const std::string& key) const {
        if (value->is<picojson::object>() && value->get<picojson::object>().count(key)) {
            return JsonAccessor(value->get<picojson::object>().at(key));
        }
        return JsonAccessor(picojson::value());
    }

    JsonAccessor operator[](size_t index) const {
        if (value->is<picojson::array>() && index < value->get<picojson::array>().size()) {
            return JsonAccessor(value->get<picojson::array>().at(index));
        }
        return JsonAccessor(picojson::value());
    }

    bool isArray() const {
        return value->is<picojson::array>();
    }

    // 检查是否为对象
    bool isObject() const {
        return value->is<picojson::object>();
    }

    const picojson::array& getArray() const {
        if (isArray()) {
            return value->get<picojson::array>();
        }
        return emptyArray;
    }

    std::optional<float> getFloat() const {
        if (value->is<double>()) {
            return static_cast<float>(value->get<double>());
        }
        return std::nullopt;
    }

    std::optional<int> getInt() const {
        if (value->is<double>()) {
            return static_cast<int>(value->get<double>());
        }
        return std::nullopt;
    }

    std::optional<unsigned int> getArraySize() const {
        if (value->is<picojson::array>()) {
            return static_cast<unsigned int>(value->get<picojson::array>().size());
        }
        return std::nullopt;
    }

    // we can not make the order!
    std::vector<std::pair<std::string, JsonAccessor>> getObjectItems() const {
        std::vector<std::pair<std::string, JsonAccessor>> items;
        if (isObject()) {
            for (const auto& item : value->get<picojson::object>()) {
                std::cout << item.first << std::endl;
                items.emplace_back(item.first, JsonAccessor(item.second));
            }
        }
        return items;
    }
};


static inline bool isSubstring(const std::string& a, const std::string& b) {
    return b.find(a) != std::string::npos;
}

static inline bool isSubstring(const char* a, const char* b) {
    if (a == nullptr || b == nullptr) {
        return false;
    }
    return strstr(b, a) != nullptr;
}

static inline JitModelInfo_t read_from_config(const std::string& config_path) {
    JitModelInfo_t info;
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << config_path << std::endl;
        return info;
    }
    std::string json_str((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    picojson::value v;
    PicoJsonParse(v, json_str);
    // extract data
    JsonAccessor accessor(v);
    auto input_params  = accessor["input_params"].getArray();
    auto input_buffers = accessor["input_buffers"].getArray();

    for (const auto& each_param : input_params) {
        auto param = JsonAccessor(each_param).getString();
        if (param.has_value()) {
            info.input_params.push_back(param.value());
        }
    }

    for (const auto& each_buffer : input_buffers) {
        auto buffer = JsonAccessor(each_buffer).getString();
        if (buffer.has_value()) {
            info.input_buffers.push_back(buffer.value());
        }
    }

    auto user_inputs   = accessor["user_inputs"].getArray();
    auto user_outputs  = accessor["user_outputs"].getArray();

    for (const auto& each_input : user_inputs) {
        auto input = JsonAccessor(each_input).getString();
        if (input.has_value()) {
            info.user_inputs.push_back(input.value());
        }
    }

    for (const auto& each_output : user_outputs) {
        auto output = JsonAccessor(each_output).getString();
        if (output.has_value()) {
            info.user_outputs.push_back(output.value());
        }
    }

    auto output_params  = accessor["output_params"].getArray();
    auto output_buffers = accessor["output_buffers"].getArray();

    for (const auto& each_param : output_params) {
        auto param = JsonAccessor(each_param).getString().value();
        auto idx = std::find(info.input_params.begin(), info.input_params.end(), param);
        if (idx != info.input_params.end()) {
            info.output_params.push_back(std::distance(info.input_params.begin(), idx));
        }
    }

    for (const auto& each_buffer : output_buffers) {
        auto buffer = JsonAccessor(each_buffer).getString().value();
        auto idx = std::find(info.input_buffers.begin(), info.input_buffers.end(), buffer);
        if (idx != info.input_buffers.end()) {
            info.output_buffers.push_back(std::distance(info.input_buffers.begin(), idx));
        }
    }
    return info;
}