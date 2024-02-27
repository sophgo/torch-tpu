#include <thread>
#include <atomic>
#include <iostream>

class DummyDaemon
{
public:
    DummyDaemon(){
        // std::cout << "DummyDamon thread init \n";
        _run = true;
        _daemon_thread = std::thread([this](){while(_run) {}});
    }
    DummyDaemon(const DummyDaemon&) = delete;
    DummyDaemon(const DummyDaemon&&) = delete;

    ~DummyDaemon(){
        _run = false;
        // std::cout << "DummyDamon thread finish\n";
        _daemon_thread.join();
    }
private:
    std::thread _daemon_thread;
    std::atomic<bool> _run;
};

DummyDaemon dummyDaemon = DummyDaemon();