
#include "help.h"
#include "compile.h"



int main(int argc, char** argv) {

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <bmodel_path> <info_path> <pt_path>\n";
        return -1;
    }

    torch::manual_seed(42);
    // std::string bmodel_path = "/workspace/newcompile/models2/vgg/vgg_8.bmodel";
    // std::string info_path   = "/workspace/newcompile/models2/vgg/info.json";
    // std::string pt_path     = "/workspace/newcompile/models2/vgg/vgg16.pt";
    std::string bmodel_path = argv[1];
    std::string info_path   = argv[2];
    std::string pt_path     = argv[3];
    TPUCompile model(bmodel_path, info_path, pt_path, /* device_id=*/ 0, /* chip=*/"bm1684x");
    torch::optim::SGD optimizer(model.parameters(), 0.001);
    auto inputs  = torch::randn({8, 3, 224, 224});
    auto targets = torch::randint(0, 1000, {8});
    for(int i = 0; i < 4; i++){
        optimizer.zero_grad();
        auto res     = model.forward(inputs, targets);
        auto loss    = res[0];
        res = model.forward(inputs, targets);
        loss = res[0];
        // loss.backward(); backward has been in bmodel, so we do not need to call it again. and this will only backward one tensor
        optimizer.step();
    }
    std::cout << "loss : " << loss.item<float>() << std::endl;
    model.cpu();
    model.save("./test.pt");
    return 0;
}