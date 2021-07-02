#ifndef GENERATOR_TRICK_H
#define GENERATOR_TRICK_H

#include <deque>
#include <functional>
#include <memory>

template <typename T>
class GeneratorTrick
{
public:
    GeneratorTrick()
    {
        generator_stop = false;
    }
    virtual ~GeneratorTrick(){};

    std::shared_ptr<T> next()
    {
        if (current == nullptr) {
            start();
        }
        while (true) {
            if (generator_stop) {
                break;
            }
            current();
            if (yield_queue.size() > 0) {
                std::shared_ptr<T> front
                    = std::make_shared<T>(yield_queue.front());
                yield_queue.pop_front();
                return front;
            }
        }
        return nullptr;
    }

    void yield(T value)
    {
        yield_queue.push_back(value);
    }

protected:
    bool generator_stop;
    std::function<void()> current;

private:
    std::deque<T> yield_queue;
    virtual void start() = 0;
};

#endif
