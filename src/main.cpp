#include "common.hpp"

class KullaContyIBL:public gl_app_t{
public:
    using gl_app_t::gl_app_t;
private:
    void initialize() override{

    }

    void frame() override{

    }

    void destroy() override{

    }

private:

};

int main(){
    KullaContyIBL(window_desc_t{
        .size = {1280,720},
        .title = "SkyRenderer",
        .resizeable = false,
        .multisamples = 4,
    }).run();
}