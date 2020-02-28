#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>

typedef enum
{
    TILT_STATUS_STOPPED = 0x00,
    TILT_STATUS_LIMIT = 0x01,
    TILT_STATUS_MOVING = 0x04,
} freenect_tilt_status_code;

typedef struct
{
    int16_t accelerometer_x;
    int16_t accelerometer_y;
    int16_t accelerometer_z;
    int8_t tilt_angle;
    freenect_tilt_status_code tilt_status;
} freenect_raw_tilt_state;

int main(int argc, char **argv)
{
    FILE *fp;
    fp = fopen(argv[1], "r");

    freenect_raw_tilt_state state;
    fread(&state, sizeof(state), 1, fp);

    double x = state.accelerometer_x;
    double y = state.accelerometer_y;
    double z = state.accelerometer_z;
    double angle = state.tilt_angle;
    std::cout << x << "  " << y << "  " << z << "  " << angle << "  " << state.tilt_status << std::endl;

    std::ofstream myfile(argv[2]);
    if (myfile.is_open())
    {
        myfile << std::fixed << std::setprecision(8) << x << "," << y << "," << z << "," << angle;
        myfile.close();
    }

    return 0;

}
