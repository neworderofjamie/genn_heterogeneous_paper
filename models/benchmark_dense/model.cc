#include <fstream>

#include "modelSpec.h"


class Blank : public NeuronModels::Base
{
public:
    DECLARE_SNIPPET(Blank);

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_SNIPPET(Blank);

class StaticPulseHalf : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(StaticPulseHalf);

    SET_VARS({{"g", "scalar", "half", VarAccess::READ_ONLY}});

    SET_PRE_SPIKE_SYN_CODE("addToPost(g);\n");
};

IMPLEMENT_SNIPPET(StaticPulseHalf);

void modelDefinition(ModelSpec &model)
{
    model.setDT(1.0);
    model.setSeed(1234);
    model.setName("benchmark_dense");
    model.setDefaultVarLocation(VarLocation::DEVICE);
    model.setDefaultSparseConnectivityLocation(VarLocation::DEVICE);
    model.setTimingEnabled(true);
    
    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // Create poisson and blank 
    auto *poisson = model.addNeuronPopulation<NeuronModels::Poisson>("Poisson", 6000, {{"rate", 100.0}}, 
                                                                     {{"timeStepToSpike", 0.0}});
    auto *output = model.addNeuronPopulation<Blank>("Output", 100000);
    
    model.addSynapsePopulation(
        "Syn", SynapseMatrixType::DENSE, poisson, output,
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, {{"g", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
}

void simulate(const ModelSpec &model, Runtime::Runtime &runtime)
{
    runtime.allocate();
    runtime.initialize();
    runtime.initializeSparse();
    
    const auto startTime = std::chrono::high_resolution_clock::now();
    while(runtime.getTimestep() < 1000) {
        runtime.stepTime();
    }
    std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - startTime;
    std::cout << "Init time:" << runtime.getInitTime() << std::endl;
    std::cout << "Total simulation time:" << duration.count() << " seconds" << std::endl;
    std::cout << "\tPresynaptic update time:" << runtime.getPresynapticUpdateTime() << std::endl;
}

