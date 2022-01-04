/*
Install opencv FIRST

*/

#include <iostream>
#include <string>
#include <sys/stat.h>

using namespace std;

int main() {
    struct stat sb = { 0 };
    string path = "PATH" ; //  img file structure
    string work_dir;

    // automatically file structure
    for (int i=0; i<10; i++)
    {
        work_dir = path;
        work_dir += '/';
        work_dir += i+'0';

        cout << "data path: " << work_dir << endl;
        if(stat(work_dir.c_str(), &sb) == -1 ) // if result of stat is -1, It means "NOT EXIST path folder"
        {
            mkdir(work_dir.c_str(), 0700);
        } 
    }
}
