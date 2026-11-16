#include <iostream>
#include "source/encoder/encOptions.h"
#include "source/commonlib/utility.h"

int main(int argc, char** argv) {
    char filein[1000] = {0};
    char fileout[1000] = {0};
    int rows = 0, cols = 0, f = 0, s = 0, t = 0;
    int ret = readOptions(argc, argv, filein, fileout, &rows, &cols, &f, &s, &t);
    std::cout << "ret=" << ret << " filein=[" << filein << "] fileout=[" << fileout << "] rows=" << rows << " cols=" << cols << " f=" << f << " s=" << s << " t=" << t << std::endl;
    FILE* fp = openFile(filein, "rb");
    std::cout << "openFile fp=" << (fp ? "SUCCESS" : "NULL") << std::endl;
    if (fp) fclose(fp);
    return 0;
}
