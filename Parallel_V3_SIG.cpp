// Version_3 of RDM  + working on calculating the significance(each unique pair distance).//
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <typeinfo>
#include <string>
#include <sstream>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <tuple>
#include <fstream>
#include <set>
#include <unistd.h>
#include <ctime>
#include <chrono>
#include <random>
#define Master 0

using namespace std;

int line_att_counter(char *fileName, int &num_att);
void start_end_line_calculator(int *start_line, int *end_line, const int numtasks, const int numPointsTotall);
ostream &operator<<(std ::ostream &os, const vector<string> &v);
void read_initial_clusters(const string fileName, const int *start_line, const int *end_line, const int k, const int rank, vector<vector<vector<string>>> &points_in_clusters);
ostream &operator<<(std ::ostream &os, const vector<int> &v);
void printer(const vector<vector<vector<string>>> vvv, const int rank);
unordered_map<int, unordered_map<int, vector<string>>> points_to_attributes(vector<vector<vector<string>>> &points_in_clusters, int k);
string vec_to_str(const vector<string> &v);
vector<string> strToVec(const string mode_unit_str);
string unMapToStr(unordered_map<int, vector<string>> &unMap);

void cstrToUnmap(const char *global_modes_cs, unordered_map<int, vector<string>> &global_modes);
float distance_calculate_1(const vector<string> &point, const vector<string> &mode);
vector<vector<vector<string>>> re_cluster(vector<vector<vector<string>>> &points_in_clusters, const unordered_map<int, vector<string>> &global_modes, vector<vector<vector<string>>> &points_to_be_added, unordered_map<int, vector<int>> &frequencies, unordered_map<int, vector<int>> &curr_no_of_points);
float E_local_calculator(const vector<vector<vector<string>>> &points_in_clusters, const unordered_map<int, vector<string>> &global_modes);
string vec_to_str_line(const vector<string> &v);
float distance_calculate_2(const vector<string> &point, const vector<string> &mode, const int num_mode, unordered_map<int, vector<int>> &frequencies,
                           unordered_map<int, vector<int>> &curr_no_of_points);

unordered_map<string, int> histogram_of_one_att_builder(const vector<string> &v);
unordered_map<int, unordered_map<int, unordered_map<string, int>>> one_proc_freq_builder(unordered_map<int, unordered_map<int, vector<string>>> &attributes);
string unMap_3d_ToStr(unordered_map<int, unordered_map<int, unordered_map<string, int>>> &unMap);
string convertToString(char *a);
tuple<string, int> find_mode_partial_2(const multimap<pair<int, int>, pair<string, int>> &mmp, const int clust_num, const int att_num);
unordered_map<int, vector<string>> find_mode_global(const int k, const int num_att, multimap<pair<int, int>, pair<string, int>> &mmp, unordered_map<int, vector<int>> &frequencies);
void curr_no_points_builder(const int *nums_in_clus_glbl_1, const int num_att, const int k, unordered_map<int, vector<int>> &curr_no_of_points);
multimap<pair<int, int>, pair<string, int>> mmp_builder2(const char *all_proc_freq_cs, int all_proc_freq_cs_size);

vector<vector<vector<string>>> all_points_in_clusters_extractor(char *result_cs, int result_cs_size, int k);
template <class T>
ostream &operator<<(ostream &os, const vector<T> &v);
template <class T>
T average_calculater(vector<T> &v);
unordered_map<int, vector<string>> points_to_attribute0(vector<vector<vector<string>>> &points_in_clusters);
unordered_map<int, unordered_map<string, int>> freq_table0_builder(unordered_map<int, vector<string>> &attribute0);
void pre_rec_f1_acc_v2(int ext_itr, int num_points, vector<vector<vector<string>>> &points_in_clusters, vector<float> &pre_w, vector<float> &rec_w, vector<float> &f1_w,
                       vector<float> &pre_M, vector<float> &rec_M, vector<float> &f1_M, vector<float> &acc);
void rd_read_initial_clusters(const string fileName, const int *start_line, const int *end_line, const int k, const int rank, vector<vector<vector<string>>> &points_in_clusters);

void read_initial_clusters_nonRD(const string fileName, const int *start_line, const int *end_line, const int k, const int rank, vector<vector<vector<string>>> &points_in_clusters);
unordered_map<int, vector<string>> points_to_attributes_nonClust(vector<vector<vector<string>>> &points_in_clusters, int k);
unordered_map<int, unordered_map<string, int>> one_proc_freq_builder_nonClust(unordered_map<int, vector<string>> &attributes_nonClust);
string unMap_3d_ToStr_nonClust(unordered_map<int, unordered_map<string, int>> &unMap);
multimap<int, pair<string, int>> mpp_builder(const char *all_proc_freq_nonClust_cs, int all_proc_freq_nonClust_cs_size, const int rank);
unordered_map<int, unordered_map<string, int>> freq_table_global_builder(const int num_att, multimap<int, pair<string, int>> &mpp);
multimap<float, vector<string>, greater<float>> density_builder_dmm(vector<vector<string>> &rdm_point_s_lc, unordered_map<int, unordered_map<string, int>> &freq_table_global);
vector<vector<vector<string>>> re_cluster(int k, const unordered_map<int, vector<string>> &global_modes,
                                          unordered_map<int, vector<int>> &frequencies, unordered_map<int, vector<int>> &curr_no_of_points, vector<vector<float>> &k_distXdens,
                                          const multimap<float, vector<string>, greater<float>> &cpyOfdens_point, const int rank);
vector<string> EmpClustHandler(const int k, vector<vector<float>> &k_distXdens,
                               const multimap<float, vector<string>, greater<float>> cpyOfdens_point, const int emptyClusterIndex, float &myLocalMax, int &myMaxDistXDensIndex, float *all_maxes, const int rank, const int numtasks, vector<float> &min_distXdens);
float distance_calculate_1_v2(const vector<string> &point1, const vector<string> &point2);
unordered_map<int, vector<string>> find_mode_global_partial(const int k, const int num_att, multimap<pair<int, int>, pair<string, int>> &mmp);
void rd_read_initial_clusters(const string fileName, const int *start_line, const int *end_line, const int rank, vector<vector<string>> &rdm_point_s_lc);
unordered_map<int, vector<string>> points_to_attributes_nonClust(vector<vector<string>> &rdm_point_s_lc, int k);

map<int, vector<string>> points_to_attributes_map(vector<vector<string>> &point_s);
map<int, map<string, int>> freq_table_builder_map(map<int, vector<string>> &attributes);
map<string, int> histogram_of_one_att_builder_map(const vector<string> &v);
vector<pair<string, string>> pairsOfattVals_builder(map<int, map<string, int>> &freq_table_map, int att_num);
vector<string> att_vals_builder(map<int, map<string, int>> &freq_table_map, int att_num);
string my_local_att_vals_str_builder(map<int, map<string, int>> &freq_table_map, const int num_att);
map<int, vector<string>> gl_att_vals_builder(const char *glb_att_vals_cs, const int glb_att_vals_cs_size);
void fourNums_4eachAttVal_builder(pair<string, string> &myPair, int att_num_pri, map<int, vector<string>> &glbl_unique_att_vals, int att_num_sec, vector<vector<string>> &point_s, map<int, map<pair<string, string>, map<int, map<string, vector<int>>>>> &partial_table_one_proc, string &partial_table_str);
void partial_table_builder(const int num_att, unordered_map<int, vector<pair<string, string>>> &att_pairs, vector<vector<string>> &rdm_point_s_lc, map<int, vector<string>> &glbl_unique_att_vals, map<int, map<pair<string, string>, map<int, map<string, vector<int>>>>> &partial_table_one_proc);
float dist_pair_finder_pp(map<int, map<pair<string, string>, float>> &dist_pair, vector<string> &point1, vector<string> &point2, int att_num);
float dist_calculator_sig_pp(vector<string> point1, vector<string> point2, map<int, map<pair<string, string>, float>> &dist_pair);

struct myKey
{
    int pri;
    pair<string, string> myPair;
    int sec;
    string att_val;
};

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);
    int rank, numtasks;
    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    char fileName[] = "BreastCancerDataSet.csv";
    int k = 2; // we have to clarify # of clusters.
               // CarDataSet.csv , BreastCancerDataSet , BCToy , CongressionalVotesDataSet , MushroomDataSet
               // Mushroom_diff_order  ,  CongressionalVotes_diff_order  , BreastCancer_diff_order , vote_toy , vote_toy_20
               // zoo.csv , audiology.standardized , balance-scale , letter-recognition , MushroomDataSet_512 , soybean-small

    /****line_att_counter*****/
    int num_att = 0;
    int numPointsTotal = line_att_counter(fileName, num_att); // if getting ready-made initial seeds , we have to deduct from number of atts
    if (rank == Master)
    {
        cout << "numPointsTotal: " << numPointsTotal << endl;
        cout << "num_att : " << num_att << endl;
        cout << " K : " << k << endl;
    }

    /*****read_initial_clusters*****/
    /*****Defining start_line and end_line for each processor (numPointsLocal also implicitly*****/
    int *start_line = new int[numtasks]();
    int *end_line = new int[numtasks]();

    start_end_line_calculator(start_line, end_line, numtasks, numPointsTotal);

    vector<float> pre_w;
    vector<float> rec_w;
    vector<float> f1_w;
    vector<float> pre_M;
    vector<float> rec_M;
    vector<float> f1_M;
    vector<float> acc;
    vector<float> itrNum;
    vector<int> timeElapsed;
    vector<int> timeElapsedRDM;
    vector<int> timeElapsed_process; //-time to calculate density
    vector<float> E_all;

    int total_iterations = 1;
    int ext_itr = 0;

    while (ext_itr < total_iterations)
    {

        if (rank == Master)
        {
            cout << "#####################################" << endl;
            cout << "EXTERNAL ITERATION ------ " << ext_itr + 1 << endl;
        }

        vector<vector<vector<string>>> points_in_clusters;
        points_in_clusters.resize(k); // to initialize the first dimension which is equal to the No. of clusters
        vector<vector<string>> rdm_point_s_lc;
        rd_read_initial_clusters(fileName, start_line, end_line, rank, rdm_point_s_lc); // function to read and randomize the share of each process.

        // Getting the size of each process's points and summing it up at the end to check if numPointsTotal == total_size
        int my_size = rdm_point_s_lc.size();
        int total_size = 0;

        MPI_Reduce(&my_size, &total_size, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        auto start_RDM_s = std::chrono::high_resolution_clock::now();

        /*****************************************************/
        /*************SIGNIFICANCE - dist_pair****************/
        /*****************************************************/

        // freq_table_map and att_map to use in several places
        map<int, vector<string>> attributes_map = points_to_attributes_map(rdm_point_s_lc);
        map<int, map<string, int>> freq_table_map = freq_table_builder_map(attributes_map);

        // building att_pairs : all pairs of each att
        unordered_map<int, vector<pair<string, string>>> att_pairs;
        for (size_t i = 0; i < num_att; ++i)
        {
            vector<pair<string, string>> pairs = pairsOfattVals_builder(freq_table_map, i); // every process has its own personilized list of pairs for each att
            att_pairs[i] = pairs;
        }

        /***************************************LOCAL UNIQUE ATT VALS************************************/
        /***********************Share it with other process and build glbl_att_vals**********************/
        /*******glbl_att_vals is non_unique, we should convert it to glbl_unique_att_vals****************/

        string lc_att_vals_str = my_local_att_vals_str_builder(freq_table_map, num_att);

        char *lc_att_vals_cs = new char[lc_att_vals_str.size() + 1]; // very important +1 source of whole lot of unknwn errors if not added
        strcpy(lc_att_vals_cs, lc_att_vals_str.c_str());

        /*********Gathering this lc_att_vals_cs from all processors and store it in one Cstring*********/

        int glb_att_vals_cs_size = 0;
        int count_send_5 = lc_att_vals_str.size();
        int *counts_recv_5 = new int[numtasks](); //@
        int *displs_5 = new int[numtasks]();      //@

        // adding up all the count_send of all the processors to calculate the total size of all_proc_freq_cs_size
        MPI_Allreduce(&count_send_5, &glb_att_vals_cs_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        char *glb_att_vals_cs = new char[glb_att_vals_cs_size + 1]();

        // Building counts_recv:
        MPI_Allgather(&count_send_5, 1, MPI_INT, counts_recv_5, 1, MPI_INT, MPI_COMM_WORLD);

        // Building displs

        displs_5[0] = 0;
        for (size_t i = 1; i < numtasks; ++i)
        {
            displs_5[i] = displs_5[i - 1] + counts_recv_5[i - 1];
        }

        // sending and receiving the frequency tables_nonClust for all the atts in each process(every process is going to have this data.
        MPI_Allgatherv(lc_att_vals_cs, lc_att_vals_str.size(), MPI_CHAR, glb_att_vals_cs, counts_recv_5, displs_5, MPI_CHAR, MPI_COMM_WORLD);

        delete counts_recv_5;
        delete displs_5;

        map<int, vector<string>> glbl_att_vals = gl_att_vals_builder(glb_att_vals_cs, glb_att_vals_cs_size);
        map<int, map<string, int>> freq_table_map_2 = freq_table_builder_map(glbl_att_vals); // it is needed to build the distinct att_vals (glbl_unique_att_vals)
        map<int, vector<string>> glbl_unique_att_vals;

        for (size_t i = 0; i < num_att; ++i)
        {

            vector<string> att_vals = att_vals_builder(freq_table_map_2, i);
            glbl_unique_att_vals[i] = att_vals;
        }

        delete lc_att_vals_cs;
        delete glb_att_vals_cs;

        /************************************************************************************************/
        /******************partial_table_one_proc********partial_table_str******************************/
        /************************************************************************************************/

        // att_num_pri-> pair , att_num_sec -> att_val -> p_fst, p_not_sec, cnt_fst, cnt_sec
        map<int, map<pair<string, string>, map<int, map<string, vector<int>>>>> partial_table_one_proc;
        string partial_table_str = "";

        for (size_t i = 0; i < num_att; ++i)
        { // att_num_pri

            for (size_t pair_num = 0; pair_num < att_pairs.at(i).size(); ++pair_num)
            { // all the pairs of the primary att

                for (size_t j = 0; j < num_att; ++j)
                { // att_num_sec

                    if (j != i)
                    { // to skip the primary att itself

                        fourNums_4eachAttVal_builder(att_pairs.at(i).at(pair_num), i, glbl_unique_att_vals, j, rdm_point_s_lc, partial_table_one_proc, partial_table_str);

                    } // if

                } // for-secondary att

            } // for-which pair

        } // primary att

        /******************************SENDING AND RECEIVING partial_table_str***************************/
        char *partial_table_cs = new char[partial_table_str.size() + 1]; // very important +1 source of whole lot of unknwn errors if not added
        strcpy(partial_table_cs, partial_table_str.c_str());

        /*********Gathering this partial_table_str from all processors and store it in one Cstring*********/

        int full_table_cs_size = 0;
        int count_send_6 = partial_table_str.size();
        int *counts_recv_6 = new int[numtasks]();
        int *displs_6 = new int[numtasks]();

        // adding up all the count_send of all the processors to calculate the total size of all_proc_freq_cs_size
        MPI_Allreduce(&count_send_6, &full_table_cs_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        char *full_table_cs = new char[full_table_cs_size + 1]();

        // Building counts_recv:
        MPI_Allgather(&count_send_6, 1, MPI_INT, counts_recv_6, 1, MPI_INT, MPI_COMM_WORLD);

        // Building displs
        displs_6[0] = 0;
        for (size_t i = 1; i < numtasks; ++i)
        {
            displs_6[i] = displs_6[i - 1] + counts_recv_6[i - 1];
        }

        // sending and receiving the frequency tables_nonClust for all the atts in each process(every process is going to have this data.
        MPI_Allgatherv(partial_table_cs, partial_table_str.size(), MPI_CHAR, full_table_cs, counts_recv_6, displs_6, MPI_CHAR, MPI_COMM_WORLD);

        delete counts_recv_6;
        delete displs_6;

        /*******************************************************************************************/
        /*******************************************************************************************/

        map<myKey, vector<float>> full_table;

        string pri = "";
        string pair_frs = "";
        string pair_sec = "";
        string sec = "";
        string att_val = "";
        string num = "";
        vector<float> fourNums;

        int i = 0;
        //@0@L,T%1$A*1#1#1#1!@0@L,T%1$B*0#0#1#1!@0@L,T%1$C*0#1#1#1!@0@L,T%1$D*0#1#1#1!

        while (i < full_table_cs_size)
        {

            if (full_table_cs[i] == '@')
            {
                i++;
                while (full_table_cs[i] != '@')
                {
                    pri = pri + full_table_cs[i];
                    i++;
                }

            } // if

            i++;
            while (full_table_cs[i] != ',')
            {
                pair_frs = pair_frs + full_table_cs[i];
                i++;
            }

            i++;
            while (full_table_cs[i] != '%')
            {
                pair_sec = pair_sec + full_table_cs[i];
                i++;
            }

            i++;
            while (full_table_cs[i] != '$')
            {
                sec = sec + full_table_cs[i];
                i++;
            }

            i++;
            while (full_table_cs[i] != '*')
            {
                att_val = att_val + full_table_cs[i];
                i++;
            }

            i++;
            //@0@L,T%1$A*1#1#1#1#@
            //@0@L,T%1$A*1#1#1#1#

            for (int j = 0; j < 4; ++j)
            { // four numbers
                while (full_table_cs[i] != '#')
                {
                    num = num + full_table_cs[i];
                    i++;
                }

                fourNums.push_back(stof(num));
                num = "";

                if (i != full_table_cs_size - 1)
                {
                    i++;
                }

            } // for

            struct myKey k1 = {stoi(pri), make_pair(pair_frs, pair_sec), stoi(sec), att_val};

            if (full_table.count(k1) != 0)
            { // if we already have the same key, we need to sum all the numbers up too.
                vector<float> temp = full_table.at(k1);

                for (int i = 0; i < 4; ++i)
                {
                    temp[i] = temp.at(i) + fourNums.at(i);
                }

                full_table[k1] = temp;
            }
            else
            {

                full_table[k1] = fourNums;
            }

            pri = "";
            pair_frs = "";
            pair_sec = "";
            sec = "";
            att_val = "";
            num = "";
            fourNums.clear();

            //@0@L,T%1$A*1#1#1#1#@
            //@0@L,T%1$A*1#1#1#1#
            if (full_table_cs[i] != '@')
            {
                i++;
            }

        } // most outer while

        /*if(rank == Master){

            for(auto itr = full_table.begin() ; itr != full_table.end() ; ++itr){
                cout<<"************"<<endl;
                struct myKey k = itr->first;
                cout<<k.pri<<endl;
                cout<<k.myPair.first<<" , "<<k.myPair.second<<endl;
                cout<<k.sec<<endl;
                cout<<k.att_val<<endl;
                vector<float> v = itr->second;
                for(int i = 0 ; i < 4 ; i++){
                    cout<<v.at(i)<<",";
                }
                cout<<endl;
            }

        }*/

        map<int, map<pair<string, string>, map<int, float>>> max_dist_pair;

        for (auto itr = full_table.begin(); itr != full_table.end(); ++itr)
        {

            struct myKey k = itr->first;
            int pri = k.pri;
            pair<string, string> myPair = k.myPair;
            int sec = k.sec;
            string att_val = k.att_val;
            vector<float> v = itr->second;

            float dist = (float)(v.at(0) / v.at(2)) + (float)(v.at(1) / v.at(3)) - 1.0;

            auto it1 = max_dist_pair.find(pri);
            if (it1 != max_dist_pair.end())
            {
                auto it2 = it1->second.find(make_pair(myPair.first, myPair.second));
                if (it2 != it1->second.end())
                {
                    auto it3 = it2->second.find(sec);
                    if (it3 != it2->second.end())
                    { // all these ifs are to check if we have the same complex key or not

                        float old_dist = max_dist_pair[pri][myPair][sec];

                        if (dist > old_dist)
                        {
                            max_dist_pair[pri][myPair][sec] = dist; // putting the max among the two (dist & old_dist) in the dist_pair.
                        }
                    } // if sec
                    else
                    {

                        max_dist_pair[pri][myPair][sec] = dist;

                    } // else sec
                }     // if myPair
                else
                {
                    max_dist_pair[pri][myPair][sec] = dist;

                } // else myPair
            }     // if pri
            else
            {

                max_dist_pair[pri][myPair][sec] = dist;

            } // else pri

        } // for full_table

        // map<int, map< pair<string,string> , map<int,float>>> max_dist_pair
        // calculating the distance and (significance)
        vector<float> significance;
        map<int, map<pair<string, string>, float>> dist_pair;

        for (auto itr = max_dist_pair.begin(); itr != max_dist_pair.end(); ++itr)
        {
            // float sum_sig = 0.0;
            // cout<<"FOR PRI_ATT-->"<<itr->first<<endl;
            map<pair<string, string>, map<int, float>> mpm = itr->second;

            for (auto itr1 = mpm.begin(); itr1 != mpm.end(); ++itr1)
            {
                float sum_dist = 0.0;
                map<int, float> mif = itr1->second;

                for (auto itr2 = mif.begin(); itr2 != mif.end(); ++itr2)
                {

                    sum_dist = sum_dist + itr2->second;
                }

                dist_pair[itr->first][make_pair(itr1->first.first, itr1->first.second)] = (float)sum_dist / (num_att - 1.0);
            }
        }

        auto end_RDM_s = std::chrono::high_resolution_clock::now();

        auto time_dens_start = std::chrono::high_resolution_clock::now();

        /****************************************************************************************/
        /****************Calculating the density of data objects in each process*****************/
        /************************************dens_point*****************************************/

        unordered_map<int, vector<string>> attributes_nonClust = points_to_attributes_nonClust(rdm_point_s_lc, k);                   // int: att_num
        unordered_map<int, unordered_map<string, int>> one_proc_freq_nonClust = one_proc_freq_builder_nonClust(attributes_nonClust); // int: att_num

        /******Converting this 3-d unMap to Cstring to be able to send it using MPI platform*********/
        string one_proc_freq_nonClust_str = unMap_3d_ToStr_nonClust(one_proc_freq_nonClust);
        one_proc_freq_nonClust_str = one_proc_freq_nonClust_str + '#'; // adding a # at the end of this string

        char *one_proc_freq_nonClust_cs = new char[one_proc_freq_nonClust_str.size() + 1]; // very important +1 source of whole lot of unknwn errors if not added
        strcpy(one_proc_freq_nonClust_cs, one_proc_freq_nonClust_str.c_str());

        /*********Gathering this one_proc_freq_cs from all processors and store it in one Cstring*********/
        int all_proc_freq_nonClust_cs_size = 0;
        int count_send_2 = one_proc_freq_nonClust_str.size();
        int *counts_recv_2 = new int[numtasks]();
        int *displs_2 = new int[numtasks]();

        // adding up all the count_send of all the processors to calculate the total size of all_proc_freq_cs_size
        MPI_Allreduce(&count_send_2, &all_proc_freq_nonClust_cs_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        char *all_proc_freq_nonClust_cs = new char[all_proc_freq_nonClust_cs_size + 1](); //@

        // Building counts_recv:
        MPI_Allgather(&count_send_2, 1, MPI_INT, counts_recv_2, 1, MPI_INT, MPI_COMM_WORLD);

        // Building displs
        displs_2[0] = 0;
        for (size_t i = 1; i < numtasks; ++i)
        {
            displs_2[i] = displs_2[i - 1] + counts_recv_2[i - 1];
        }

        // sending and receiving the frequency tables_nonClust for all the atts in each process(every process is going to have this data.
        MPI_Allgatherv(one_proc_freq_nonClust_cs, one_proc_freq_nonClust_str.size(), MPI_CHAR, all_proc_freq_nonClust_cs, counts_recv_2, displs_2, MPI_CHAR, MPI_COMM_WORLD);

        delete counts_recv_2;
        delete displs_2;

        multimap<int, pair<string, int>> mpp = mpp_builder(all_proc_freq_nonClust_cs, all_proc_freq_nonClust_cs_size, rank);

        unordered_map<int, unordered_map<string, int>> freq_table_global = freq_table_global_builder(num_att, mpp); // int-> num_att
        multimap<float, vector<string>, greater<float>> dens_point = density_builder_dmm(rdm_point_s_lc, freq_table_global);

        multimap<float, vector<string>, greater<float>> cpyOfdens_point = dens_point; // to be used in case of empty cluster encountering

        delete one_proc_freq_nonClust_cs;
        delete all_proc_freq_nonClust_cs;

        /***************************END***********************************************************/
        /****************Calculating the density of data objects in each process*****************/
        /****************************************************************************************/
        auto time_dens_end = std::chrono::high_resolution_clock::now();

        /*********************************************************************************************************************************/
        auto start_RDM = std::chrono::high_resolution_clock::now(); // Starting the timer

        vector<vector<string>> centers;
        int numOfclust = 0;
        /********************************************************************/
        /*******FORMING INITIAL CLUSTERS AROUND HIGHEST DENSITY POINTS*******/
        /********************************************************************/
        while (numOfclust < k)
        {

            // checking how many points have remained before processing and deletion.
            int mySize = dens_point.size();
            int size_total = 0;
            int *all_sizes = new int[numtasks]();
            MPI_Allgather(&mySize, 1, MPI_INT, all_sizes, 1, MPI_INT, MPI_COMM_WORLD);
            MPI_Allreduce(&mySize, &size_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

            /**********Active Ranks detection (Ranks that are not yet empty and still have data objects remaining***********/
            // We construct a new communicator with the active processes to do the collective communications.
            vector<int> active_ranks;
            for (int i = 0; i < numtasks; ++i)
            {
                if (all_sizes[i] != 0)
                {
                    active_ranks.push_back(i);
                }
            }

            int *active_ranks_i = &active_ranks[0]; // converting the vector<int> to array[int]
            // Construct a group containing all of the active ranks in world_group
            MPI_Group active_group;
            MPI_Group_incl(world_group, active_ranks.size(), active_ranks_i, &active_group);

            // Create a new communicator based on the group
            MPI_Comm active_comm;
            MPI_Comm_create_group(MPI_COMM_WORLD, active_group, 0, &active_comm); // creating a communicator based on the active_group.

            int active_rank = -1, active_size = -1;
            // If this rank isn't in the new communicator, it will be
            // MPI_COMM_NULL. Using MPI_COMM_NULL for MPI_Comm_rank or
            // MPI_Comm_size is erroneous

            if (MPI_COMM_NULL != active_comm)
            {
                MPI_Comm_rank(active_comm, &active_rank);
                MPI_Comm_size(active_comm, &active_size);
            }

            if (size_total != 0)
            {

                if (mySize != 0)
                { // If i am an active process

                    /************Each process sends their maximum dens to find which point has the haighest density across all the processes************/
                    vector<string> cand_pointMaxDens = dens_point.begin()->second;
                    float myLocalMaxDens = dens_point.begin()->first;

                    string cand_pointMaxDens_str = vec_to_str(cand_pointMaxDens);
                    char *cand_pointMaxDens_cs = new char[cand_pointMaxDens_str.size() + 1]();
                    strcpy(cand_pointMaxDens_cs, cand_pointMaxDens_str.c_str());

                    float *all_maxes_dens = new float[active_size](); // every active process has a maximum
                    MPI_Allgather(&myLocalMaxDens, 1, MPI_FLOAT, all_maxes_dens, 1, MPI_FLOAT, active_comm);

                    // Now every active process has all the maximums--->identifying the process with the max distXdens point from other modes that their clusters aren't empty.
                    int global_max_rank = 0;
                    float global_max_dens = all_maxes_dens[global_max_rank];
                    if (active_size != 0)
                    {
                        for (size_t i = 1; i < active_size; ++i)
                        {
                            if (all_maxes_dens[i] > global_max_dens)
                            {
                                global_max_dens = all_maxes_dens[i];
                                global_max_rank = i;
                            }
                        }
                    }

                    // pointMaxDens is sent by the process that has it to all other processes.
                    int pointMaxDens_cs_size = 0;
                    if (active_rank == global_max_rank)
                    {
                        pointMaxDens_cs_size = cand_pointMaxDens_str.size();
                    }

                    MPI_Bcast(&pointMaxDens_cs_size, 1, MPI_INT, global_max_rank, active_comm);
                    char *pointMaxDens_cs = new char[pointMaxDens_cs_size + 1]();

                    if (active_rank == global_max_rank)
                    {
                        strcpy(pointMaxDens_cs, cand_pointMaxDens_cs);
                    }

                    MPI_Bcast(pointMaxDens_cs, pointMaxDens_cs_size, MPI_CHAR, global_max_rank, active_comm);
                    string pointMaxDens_str = pointMaxDens_cs;
                    centers.push_back(strToVec(pointMaxDens_cs)); // adding the new center

                    /************Each process calculate the dist from maxDensPoint and they do a reduce to cal ave_dis************/
                    vector<float> dist_fr_cent;
                    float sum_dist_lcl = 0;
                    int cnt = 0;
                    for (auto itr = dens_point.begin(); itr != dens_point.end(); ++itr)
                    {
                        float dist = distance_calculate_1_v2(itr->second, centers.at(numOfclust)); // dist_calculator_sig_pp
                        dist_fr_cent.push_back(dist);
                        sum_dist_lcl = sum_dist_lcl + dist;
                    }

                    float sum_dist_glbl = 0;
                    MPI_Allreduce(&sum_dist_lcl, &sum_dist_glbl, 1, MPI_FLOAT, MPI_SUM, active_comm);

                    int myLocal_dens_point_size = dens_point.size();
                    int global_dens_point_size = 0; // We have to calculate this at every step because we delete points from dens_point and it changes.
                    MPI_Allreduce(&myLocal_dens_point_size, &global_dens_point_size, 1, MPI_FLOAT, MPI_SUM, active_comm);

                    float ave_dist = (float)sum_dist_glbl / (global_dens_point_size - 1);

                    /*******************DELETION*********************************************************************************/
                    int idx = 0;
                    auto itr = dens_point.begin();
                    while (itr != dens_point.end())
                    {
                        if (dist_fr_cent[idx] <= ave_dist)
                        {
                            points_in_clusters[numOfclust].push_back(itr->second);
                            itr = dens_point.erase(itr); // it deletes itr and return the itr which points to the next pair(it goes to next itr)
                        }
                        else
                        {
                            ++itr;
                        }
                        ++idx;
                    }

                    delete cand_pointMaxDens_cs;
                    delete all_maxes_dens;
                    delete pointMaxDens_cs;
                    delete all_sizes;

                } // mySize

            } // size_total != 0
            else
            {
                if (rank == Master)
                    cout << "TOTAL EMPTYNESS! Failure of the algorithm in this case" << endl;
                MPI_Finalize();
                exit(0);
            } // size_total != 0

            numOfclust++;
            //			MPI_Group_free(&world_group);
            MPI_Group_free(&active_group);

            if (active_rank != -1)
                MPI_Comm_free(&active_comm);

        } // while(numOfClust)

        /********************FINDING THE FIRST INITIAL PARTIAL MODES OF CLUSTERS**********************/
        /********* It may not include all the points***********/
        /*********************************************************************************************/
        // Checking to see if we really need to do this
        int my_size_in_clust = 0;
        int total_size_in_clust = 0;
        for (int i = 0; i < k; ++i)
        {
            my_size_in_clust = my_size_in_clust + points_in_clusters.at(i).size();
        }
        MPI_Reduce(&my_size_in_clust, &total_size_in_clust, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (total_size_in_clust != numPointsTotal)
        { // if there remains some points that are not yet clustered.

            /************attributes**************/
            unordered_map<int, unordered_map<int, vector<string>>> attributes = points_to_attributes(points_in_clusters, k); // num_clust,num_att

            /************one_proc_freq**************/
            /************each process builds a 3 dimensional unordered_map************/
            /***** unoredered_map<int,unordered_map<int,unordered_map<string,int>>> -->1st int: clust_num , 2nd int: att_num , string: att_value , 3rd int: frequency of that att****/
            unordered_map<int, unordered_map<int, unordered_map<string, int>>> one_proc_freq = one_proc_freq_builder(attributes);

            /******Converting this 3-d unMap to Cstring to be able to send it using MPI platform*********/
            string one_proc_freq_str = unMap_3d_ToStr(one_proc_freq);
            one_proc_freq_str = one_proc_freq_str + '#'; // adding a # at the end of this string

            char *one_proc_freq_cs = new char[one_proc_freq_str.size() + 1]; // very important +1 source of whole lot of unknwn errors if not added
            strcpy(one_proc_freq_cs, one_proc_freq_str.c_str());

            /*********Gathering this one_proc_freq_cs from all processors and store it in one Cstring*********/
            int all_proc_freq_cs_size = 0;
            int count_send = one_proc_freq_str.size();
            int *counts_recv = new int[numtasks]();
            int *displs = new int[numtasks]();

            // adding up all the count_send of all the processors to calculate the total size of all_proc_freq_cs_size
            MPI_Allreduce(&count_send, &all_proc_freq_cs_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            char *all_proc_freq_cs = new char[all_proc_freq_cs_size + 1]();

            // Building counts_recv:
            MPI_Allgather(&count_send, 1, MPI_INT, counts_recv, 1, MPI_INT, MPI_COMM_WORLD);

            // Building displs
            displs[0] = 0;
            for (size_t i = 1; i < numtasks; ++i)
            {
                displs[i] = displs[i - 1] + counts_recv[i - 1];
            }

            // sending and receiving the frequency tables for all the atts in all the clusters in each process(every process is going to have this data.
            MPI_Allgatherv(one_proc_freq_cs, one_proc_freq_str.size(), MPI_CHAR, all_proc_freq_cs, counts_recv, displs, MPI_CHAR, MPI_COMM_WORLD);

            /*************converting the string back to a form of clust_num, att_num , att_val , att_freq ****************/
            /*************to have ALL the data at hand to be able to calculate the Global_mode later**********************/
            /**************** Using a multimap with 2 keys <clust_num,att_num> and a pair of<att_val,freq>****************/
            /*************We need to use a multimap to be able to store more than one value for each key pair*************/
            // multimap<pair<int,int> , pair<string,int>> mmp = mmp_builder(all_proc_one_proc_freq_str);
            multimap<pair<int, int>, pair<string, int>> mmp = mmp_builder2(all_proc_freq_cs, all_proc_freq_cs_size);

            /********************FINDING REAL GLOBAL MODES PARTIAL from multimap<pair<int,int>,pair<string , int>> by all the processes********************/
            unordered_map<int, vector<string>> global_modes_partial = find_mode_global_partial(k, num_att, mmp); // int -> clust_num

            /*There's this condition in which a process doesn't have any more points of a particular cluster..in this case, we put char "%" in all the attributes of the unavailable 				cluster'mode and then ignore these characters in find_mode_partial*/
            if (global_modes_partial.size() != k)
            {

                for (size_t i = 0; i < k; i++)
                {
                    // if we have no more points left for the process
                    if (global_modes_partial.count(i) != 1)
                    {

                        for (int j = 0; j < num_att; ++j)
                        {
                            global_modes_partial[i].push_back("%");
                        }
                    }
                }
            }

            /**************Adding whatever remains to points_in_clusters based on the initial modes which itself is partial**************/
            if (dens_point.size() != 0)
            {

                vector<float> distFromCenters;
                for (auto itr = dens_point.begin(); itr != dens_point.end(); ++itr)
                {
                    for (size_t j = 0; j < global_modes_partial.size(); ++j)
                    {
                        distFromCenters.push_back(distance_calculate_1(itr->second, global_modes_partial.at(j)));
                    }

                    int minDistanceIndex = std::min_element(distFromCenters.begin(), distFromCenters.end()) - distFromCenters.begin();
                    distFromCenters.clear();
                    points_in_clusters[minDistanceIndex].push_back(itr->second);
                }

            } // if

            delete one_proc_freq_cs;
            delete counts_recv;
            delete displs;
            delete all_proc_freq_cs;

        } // if there remains some points that are not yet clustered.

        auto end_RDM = std::chrono::high_resolution_clock::now(); // Ending the timer

        /*************###DECLARATIONS###*************/
        auto start = std::chrono::high_resolution_clock::now(); // Starting the timer

        char *all_proc_freq_cs;
        char *one_proc_freq_cs;
        int *counts_recv;
        int *displs;

        float E_total = 0.0;
        vector<float> E_history; // To be able to escape an infinite loop where we jump between repeated amounts of E andas a reason never converge
        bool condition = true;
        int iteration = 0;

        while (condition)
        {

            /************attributes**************/
            unordered_map<int, unordered_map<int, vector<string>>> attributes = points_to_attributes(points_in_clusters, k); // num_clust,num_att

            /************one_proc_freq**************/
            /************each process builds a 3 dimensional unordered_map************/
            /***** unoredered_map<int,unordered_map<int,unordered_map<string,int>>> -->1st int: clust_num , 2nd int: att_num , string: att_value , 3rd int: frequency of that att****/
            unordered_map<int, unordered_map<int, unordered_map<string, int>>> one_proc_freq = one_proc_freq_builder(attributes);

            /******Converting this 3-d unMap to Cstring to be able to send it using MPI platform*********/
            string one_proc_freq_str = unMap_3d_ToStr(one_proc_freq);
            one_proc_freq_str = one_proc_freq_str + '#'; // adding a # at the end of this string

            one_proc_freq_cs = new char[one_proc_freq_str.size() + 1]; // very important +1 source of whole lot of unknwn errors if not added
            strcpy(one_proc_freq_cs, one_proc_freq_str.c_str());

            /*********Gathering this one_proc_freq_cs from all processors and store it in one Cstring*********/
            int all_proc_freq_cs_size = 0;
            int count_send = one_proc_freq_str.size();
            counts_recv = new int[numtasks]();
            displs = new int[numtasks]();

            // adding up all the count_send of all the processors to calculate the total size of all_proc_freq_cs_size
            MPI_Allreduce(&count_send, &all_proc_freq_cs_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            all_proc_freq_cs = new char[all_proc_freq_cs_size + 1]();

            // Building counts_recv:
            MPI_Allgather(&count_send, 1, MPI_INT, counts_recv, 1, MPI_INT, MPI_COMM_WORLD);

            // Building displs
            displs[0] = 0;
            for (size_t i = 1; i < numtasks; ++i)
            {
                displs[i] = displs[i - 1] + counts_recv[i - 1];
            }

            // sending and receiving the frequency tables for all the atts in all the clusters in each process(every process is going to have this data.
            MPI_Allgatherv(one_proc_freq_cs, one_proc_freq_str.size(), MPI_CHAR, all_proc_freq_cs, counts_recv, displs, MPI_CHAR, MPI_COMM_WORLD);

            /*************converting the string back to a form of clust_num, att_num , att_val , att_freq ****************/
            /*************to have ALL the data at hand to be able to calculate the Global_mode later**********************/
            /**************** Using a multimap with 2 keys <clust_num,att_num> and a pair of<att_val,freq>****************/
            /*************We need to use a multimap to be able to store more than one value for each key pair*************/
            multimap<pair<int, int>, pair<string, int>> mmp = mmp_builder2(all_proc_freq_cs, all_proc_freq_cs_size);

            /********************FINDING REAL GLOBAL MODES from multimap<pair<int,int>,pair<string , int>> by all the processes********************/
            unordered_map<int, vector<int>> curr_no_of_points;
            unordered_map<int, vector<int>> frequencies;
            unordered_map<int, vector<string>> global_modes = find_mode_global(k, num_att, mmp, frequencies); // int -> clust_num

            /****TO CALCULATE CURR_NO_POINTS CORRECTLY****/
            int *nums_in_clus_lcl_1 = new int[k]();
            for (size_t num_clust = 0; num_clust < k; ++num_clust)
            {

                nums_in_clus_lcl_1[num_clust] = points_in_clusters.at(num_clust).size();
            }

            int *nums_in_clus_glbl_1 = new int[k](); // does an MPI_Allreduce on an array elements individually and produces the results
            MPI_Allreduce(nums_in_clus_lcl_1, nums_in_clus_glbl_1, k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

            curr_no_points_builder(nums_in_clus_glbl_1, num_att, k, curr_no_of_points);

            /*There's this condition in which a process doesn't have any more points of a particular cluster. In this case, we put char "%" in all
            the attributes of the unavailable cluster'mode and then ignore these characters in find_mode_partial*/

            if (global_modes.size() != k)
            {

                for (size_t i = 0; i < k; i++)
                {
                    // if we have no more points left for the process
                    if (global_modes.count(i) != 1)
                    {

                        // cout<<"HI FROM RANK : "<<rank<<" IN KEY : "<<i<<endl;
                        for (int j = 0; j < num_att; ++j)
                        {
                            global_modes[i].push_back("%");
                            frequencies[i].push_back(0);
                            curr_no_of_points[i].push_back(0);
                        }
                    }
                }
            }

            /*******RE_CLUSTERING********/
            vector<vector<vector<string>>> new_points_in_clusters;
            new_points_in_clusters.resize(k);

            int num_points_lcl = cpyOfdens_point.size();
            vector<vector<float>> k_distXdens;
            k_distXdens.resize(num_points_lcl); // For every point, we have "k" numbers or the distances from each of our k centorids. [num_points_lcl X k]

            new_points_in_clusters =
                re_cluster(k, global_modes, frequencies, curr_no_of_points, k_distXdens, cpyOfdens_point, rank);

            points_in_clusters.clear();
            points_in_clusters.resize(k);
            points_in_clusters = new_points_in_clusters;

            /********************************EMPTY CLUSTER CHECKING**********************************/
            /****************************************************************************************/
            int *nums_in_clus_lcl = new int[k]();
            for (size_t num_clust = 0; num_clust < k; ++num_clust)
            {

                nums_in_clus_lcl[num_clust] = points_in_clusters.at(num_clust).size();
            }

            int *nums_in_clus_glbl = new int[k]();
            MPI_Allreduce(nums_in_clus_lcl, nums_in_clus_glbl, k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

            bool emptyClusterExist = false;
            int emptyClusterIndex;
            for (int i = 0; i < k; ++i)
            {
                if (nums_in_clus_glbl[i] == 0)
                {
                    emptyClusterIndex = i;
                    emptyClusterExist = true;
                }
            }
            if (rank == Master && emptyClusterExist == true)
            {
                cout << "\n---Cluster No." << emptyClusterIndex << " is Empty---\n"
                     << endl;
            }

            delete nums_in_clus_lcl;
            delete nums_in_clus_glbl;
            /********************************EMPTY CLUSTER CHECKING**********************************/
            /********************************EMPTY CLUSTER HANDELING*********************************/

            while (emptyClusterExist)
            {

                vector<float> min_distXdens; // we get the min distXdens for every point and then find the point with the  max distXdens [num_points]
                float myLocalMax = 0.0;
                int myMaxDistXDensIndex = 0;
                float *all_maxes = new float[numtasks](); // every process has a maximum
                vector<string> modeForEmp = EmpClustHandler(k, k_distXdens, cpyOfdens_point, emptyClusterIndex, myLocalMax, myMaxDistXDensIndex, all_maxes, rank, numtasks, min_distXdens);

                global_modes[emptyClusterIndex] = modeForEmp;

                if (rank == Master)
                {

                    cout << "NEW centroid for empty cluster No." << emptyClusterIndex << " :  " << modeForEmp << endl;
                    cout << "Same itr:  " << iteration + 1 << " New Modes :" << endl;
                    for (int i = 0; i < k; ++i)
                    {
                        cout << "Clust-" << i << ": " << global_modes.at(i);
                    }
                }

                /***************RECLUSTER AGAIN WITH THE NEW MODE IN THE SAME ITERATION****************/
                vector<vector<vector<string>>> new_points_in_clusters;
                new_points_in_clusters.resize(k);

                int num_points_lcl = cpyOfdens_point.size();
                vector<vector<float>> k_distXdens;
                k_distXdens.resize(num_points_lcl); // For every point, we have "k" numbers or the distances from each of our k centorids. [num_points_lcl X k]

                new_points_in_clusters =
                    re_cluster(k, global_modes, frequencies, curr_no_of_points, k_distXdens, cpyOfdens_point, rank);

                points_in_clusters.clear();
                points_in_clusters.resize(k);
                points_in_clusters = new_points_in_clusters;

                /*************Checking again************/
                /********************************EMPTY CLUSTER CHECKING**********************************/
                /****************************************************************************************/

                nums_in_clus_lcl = new int[k]();
                for (size_t num_clust = 0; num_clust < k; ++num_clust)
                {

                    nums_in_clus_lcl[num_clust] = points_in_clusters.at(num_clust).size();
                }

                nums_in_clus_glbl = new int[k]();
                MPI_Allreduce(nums_in_clus_lcl, nums_in_clus_glbl, k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

                vector<int> emptyClustersIndex;
                for (int i = 0; i < k; ++i)
                {
                    if (nums_in_clus_glbl[i] == 0)
                    {
                        emptyClustersIndex.push_back(i);
                    }
                }
                if (emptyClustersIndex.size() == 0)
                {

                    emptyClusterExist = false;
                    if (rank == Master)
                        cout << "SOLVED THE EMPTY CLUST PROB" << endl;
                }

                delete nums_in_clus_lcl;
                delete nums_in_clus_glbl;

            } // while empty

            float E_local = E_local_calculator(points_in_clusters, global_modes);
            MPI_Allreduce(&E_local, &E_total, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); // MPI_Allreduce is equal to : MPI_Reduce and then an all-to-all broadcast

            /****************************************************/
            E_history.push_back(E_total);

            // To be able to escape an infinite loop where we jump between repeated amounts of E and as a reason never converge
            for (size_t i = 0; i < E_history.size() - 1; ++i)
            {
                if (E_total == E_history.at(i))
                {
                    condition = false;
                }
            }
            if (rank == 0)
            {
                cout << "-----------AFTER " << iteration + 1 << " iterations---------------------------E===" << E_total << endl;
            }

            iteration++;

            delete all_proc_freq_cs;
            delete one_proc_freq_cs;
            delete counts_recv;
            delete displs;

        } // end of while

        auto finish = std::chrono::high_resolution_clock::now();

        /**********************************all_points_in_clusters_str**************************************************/
        /************************************building a process's points_in_clustes_str********************************/
        /*****************************************To calculate acc , pre , rec ****************************************/

        string points_in_clusters_str = "";

        for (size_t clust_num = 0; clust_num < k; clust_num++)
        {
            string cluster_points_str = "";

            if (points_in_clusters[clust_num].size() != 0)
            {

                for (size_t j = 0; j < points_in_clusters[clust_num].size(); j++)
                { // point_num

                    cluster_points_str = cluster_points_str + vec_to_str(points_in_clusters.at(clust_num).at(j)) + "#";
                } // all the points related to cluster "clust_num" are collected in points_in_clusters_str for all the processes

                points_in_clusters_str = points_in_clusters_str + "@" + to_string(clust_num) + "@" + cluster_points_str;

            } // if

        } // for(clust_num)

        int count_send_1 = 0;
        int *counts_recv_1 = new int[numtasks]();
        int *displs_1 = new int[numtasks]();
        char *all_points_in_clusters_cs; // only has been newed in Master
        count_send_1 = points_in_clusters_str.size();
        // building counts_recv_1 in Master
        MPI_Gather(&count_send_1, 1, MPI_INT, counts_recv_1, 1, MPI_INT, Master, MPI_COMM_WORLD);

        int all_points_in_clusters_cs_size = 0;
        if (rank == Master)
        {

            displs_1[0] = 0;
            for (size_t i = 1; i < numtasks; i++)
            {
                displs_1[i] = counts_recv_1[i - 1] + displs_1[i - 1];
                all_points_in_clusters_cs_size = all_points_in_clusters_cs_size + counts_recv_1[i - 1];
            }

            all_points_in_clusters_cs_size = all_points_in_clusters_cs_size + counts_recv_1[numtasks - 1]; // adding the last process cluster_points_str's size to buf_recv_size
            all_points_in_clusters_cs = new char[all_points_in_clusters_cs_size + 1];

        } // Master

        // Master gathers a string containing all the points in all the clusters from other processes and store it in buf_recv
        MPI_Gatherv(points_in_clusters_str.c_str(), points_in_clusters_str.size(), MPI_CHAR, all_points_in_clusters_cs, counts_recv_1, displs_1, MPI_CHAR, Master, MPI_COMM_WORLD);

        vector<vector<vector<string>>> all_points_in_clusters;

        if (rank == Master)
        {

            all_points_in_clusters.resize(k);
            all_points_in_clusters = all_points_in_clusters_extractor(all_points_in_clusters_cs, all_points_in_clusters_cs_size, k);

        } // Master

        /*************************************************/
        if (rank == Master)
        {
            cout << "\nParallel V3---FINAL RESULT IS:" << endl;
            cout << "E_history :" << E_history;
            cout << "E = " << E_total << endl;
            cout << "after " << iteration << " iterations of Only clustering" << endl;

            auto time = std::chrono::duration_cast<std::chrono::milliseconds>((finish - start) + (time_dens_end - time_dens_start) + (end_RDM_s - start_RDM_s) + (end_RDM - start_RDM)).count();
            auto timeRDM = std::chrono::duration_cast<std::chrono::milliseconds>((end_RDM_s - start_RDM_s) + (end_RDM - start_RDM) + (time_dens_end - time_dens_start)).count();
            auto time_process = std::chrono::duration_cast<std::chrono::milliseconds>((finish - start)).count();

            cout << "after " << time << " ms\n";
            pre_rec_f1_acc_v2(ext_itr + 1, numPointsTotal, all_points_in_clusters, pre_w, rec_w, f1_w, pre_M, rec_M, f1_M, acc);
            itrNum.push_back(iteration);
            timeElapsed.push_back(time);
            timeElapsedRDM.push_back(timeRDM);
            timeElapsed_process.push_back(time_process);

            E_all.push_back(E_total);

            delete all_points_in_clusters_cs;
        }

        ext_itr++;

        delete counts_recv_1;
        delete displs_1;

    } // most external while

    if (rank == Master)
    {
        cout << endl;
        cout << "****************************************" << endl;
        cout << "*************General Result*************" << endl;
        cout << "**************Parallel V3***************" << endl;
        cout << "**************" << fileName << "***************" << endl;
        cout << "After " << total_iterations << " External iteration :\n"
             << endl;
        cout << endl;
        cout << "Ave Elapsed Time   = " << average_calculater(timeElapsed) << " ms" << endl
             << endl;
        cout << "Ave (RDM-SIG) Time =" << average_calculater(timeElapsedRDM) << " ms" << endl;
        cout << "Ave PROCESS Elapsed Time   = " << average_calculater(timeElapsed_process) << " ms" << endl
             << endl;

        cout << "Ave Accuracy   = " << average_calculater(acc) << endl;
        cout << "Ave Precision (M)   = " << average_calculater(pre_M) << endl;
        cout << "Ave Recall (M)   = " << average_calculater(rec_M) << endl;
        cout << endl;
        cout << endl;
        cout << "Ave internal iteration No.   = " << average_calculater(itrNum) << endl;

        cout << "Ave E = " << average_calculater(E_all) << endl;
        cout << endl;
    }

    delete start_line;
    delete end_line;

    MPI_Group_free(&world_group);
    MPI_Finalize();

    return 0;

} // main()

/*********************************************************************/
/*****************************FUNCTIONS*******************************/
/*********************************************************************/

int line_att_counter(char *fileName, int &num_att)
{

    FILE *fp;
    fp = fopen(fileName, "r");

    /*****num_att*****/
    char line[200];
    fgets(line, 200, fp);

    int linesize = strlen(line);

    int numComma = 0;
    for (int i = 0; i < linesize; i++)
    {
        if (line[i] == ',')
            numComma++;
    }
    num_att = numComma; // one-less-att-> + 1; because of the initial seed , we have deducted another one.

    int numPointsTotal = 0;
    fseek(fp, 0, SEEK_SET);
    char ch;
    while (!feof(fp))
    {
        ch = fgetc(fp);
        if (ch == '\n')
        {
            numPointsTotal++;
        }
    } // while

    fclose(fp);

    return numPointsTotal;
}

void start_end_line_calculator(int *start_line, int *end_line, const int numtasks, const int numPointsTotal)
{

    int numPointsLocal = numPointsTotal / numtasks;
    int remainder = numPointsTotal % numtasks;

    start_line[0] = 0;
    for (size_t i = 1; i < numtasks; ++i)
    {
        start_line[i] = start_line[i - 1] + numPointsLocal;
    }

    end_line[0] = numPointsLocal;
    for (size_t i = 1; i < numtasks; ++i)
    {

        if (i == numtasks - 1) // last process
            end_line[i] = end_line[i - 1] + numPointsLocal + remainder;
        else
            end_line[i] = end_line[i - 1] + numPointsLocal;

    } // for
}

void read_initial_clusters(const string fileName, const int *start_line, const int *end_line, const int k, const int rank, vector<vector<vector<string>>> &points_in_clusters)
{

    string line;
    ifstream myfile(fileName, ios::in);

    vector<string> point;

    if (!myfile.is_open())
    {
        cout << "Failed to open" << endl;
        exit(0);
    }

    int my_start_line = start_line[rank];
    int my_end_line = end_line[rank];
    int numPointsLocal = my_end_line - my_start_line;

    // Defining the size of inner clusters within each process
    // e.g: 10 points - 3 clusters => 3 , 3 , 4
    // e.g: 2 points - 3 clusters => 0 , 0 , 2 RARE CASE
    int inner_cluster_len = numPointsLocal / k;
    int inner_cluster_rm = numPointsLocal % k;
    int inner_cluster_size[k] = {};

    for (int i = 0; i < k - 1; ++i)
    {
        inner_cluster_size[i] = inner_cluster_len;
    }
    inner_cluster_size[k - 1] = inner_cluster_len + inner_cluster_rm;

    /////////////////////////////////////////////////////////////

    int num_cluster = 0;
    int cnt = 0;
    int myShare = 0;
    int line_counter = 0;

    while (getline(myfile, line))
    {

        if (myShare <= numPointsLocal) // controling the # of all the points in 1 process
        {
            if (my_start_line <= line_counter && line_counter < my_end_line)
            {
                myShare++;
                if (cnt < inner_cluster_size[num_cluster]) // arranging the points in inner clusters based on their sizes
                {

                    stringstream sline(line);
                    string word;
                    while (getline(sline, word, ','))
                    {
                        point.push_back(word);
                    }

                    points_in_clusters[num_cluster].push_back(point);
                    cnt++;
                    point.clear();
                    line_counter++;

                } // new if

                else //
                {
                    cnt = 0;
                    num_cluster++;

                    stringstream sline(line);
                    string word;
                    while (getline(sline, word, ','))
                    {
                        point.push_back(word);
                    }

                    points_in_clusters[num_cluster].push_back(point);
                    cnt++;
                    point.clear();
                    line_counter++;
                }
            }
            else
            {

                line_counter++;
            }
        }
        else
        {

            myfile.close();
        }

    } // While
}

std ::ostream &operator<<(std ::ostream &os, const vector<string> &v)
{
    for (size_t i = 0; i < v.size(); i++)
    {
        os << v[i] << " ";
    }
    os << endl;
    return os;
}

std ::ostream &operator<<(std ::ostream &os, const vector<int> &v)
{
    for (size_t i = 0; i < v.size(); i++)
    {
        os << v[i] << " ";
    }
    os << endl;
    return os;
}

void printer(const vector<vector<vector<string>>> vvv, const int rank)
{
    cout << "RANK: " << rank << " ************************" << endl;
    int cnt = 0;
    for (auto itr1 = vvv.begin(); itr1 != vvv.end(); ++itr1)
    {
        for (auto itr2 = itr1->begin(); itr2 != itr1->end(); ++itr2)
        {
            cout << cnt << " - " << *(itr2);
        }
        cnt++;
    }
}

unordered_map<int, unordered_map<int, vector<string>>> points_to_attributes(vector<vector<vector<string>>> &points_in_clusters, int k)
{

    unordered_map<int, unordered_map<int, vector<string>>> attributes; // int-> num_cluster , int-> attribute_num , for example, all the middle attributes
    // cout<<" I AM HERE AT POINTSTOATT"<<endl;

    for (size_t clus_num = 0; clus_num < k; clus_num++)
    { // iterate through cluster numbers
        for (size_t i = 0; i < points_in_clusters[clus_num].size(); ++i)
        { // iterate through points in a cluster

            vector<string> point = points_in_clusters.at(clus_num).at(i);
            for (size_t i = 1; i < point.size(); i++)
            { // i = 1 -> one-less-att

                attributes[clus_num][i - 1].push_back(point[i]); // i-1 -> one-less-att (the first one)
            }
        }
    }

    return attributes;
}

unordered_map<string, int> histogram_of_one_att_builder(const vector<string> &v)
{

    unordered_map<string, int> histogram_of_one_att;

    for (auto x : v)
    {
        histogram_of_one_att[x]++;
    }

    return histogram_of_one_att; // mode_p;//e.g "3"
}

unordered_map<int, unordered_map<int, unordered_map<string, int>>> one_proc_freq_builder(unordered_map<int, unordered_map<int, vector<string>>> &attributes)
{

    unordered_map<int, unordered_map<int, unordered_map<string, int>>> one_proc_freq; // int -> clust_num , int -> att_num

    for (size_t i = 0; i < attributes.size(); i++)
    { // clust_num
        for (size_t j = 0; j < attributes[i].size(); j++)
        { // att_num
            one_proc_freq[i][j] = histogram_of_one_att_builder(attributes.at(i).at(j));
        }
    }

    return one_proc_freq;
}

string unMap_3d_ToStr(unordered_map<int, unordered_map<int, unordered_map<string, int>>> &unMap)
{ // a function to convert the one_proc_freq to string :
  //  #clust_num#att_num#att_val#frequency#clust_num#att_num#att_val#frequency.....
    string result_str = "";

    for (size_t clust_num = 0; clust_num < unMap.size(); clust_num++)
    {
        for (size_t att_num = 0; att_num < unMap[clust_num].size(); att_num++)
        {

            unordered_map<string, int> temp = unMap[clust_num][att_num];
            string att_val;
            int freq;

            for (auto itr = temp.begin(); itr != temp.end(); itr++)
            {
                att_val = itr->first;
                freq = itr->second;
                result_str = result_str + '#' + to_string(clust_num) + '#' + to_string(att_num) + '#' + att_val + '#' + to_string(freq);
            }
        }
    }

    return result_str; // e.g #0#0#2#1#0#0#1#5#0#1#2#2#0#1#3#2#0#1#1#2#1#0#3#3#1#0#2#3#1#1#2#1#1#1#1#3#1#1#3#2
}

multimap<pair<int, int>, pair<string, int>> mmp_builder2(const char *all_proc_freq_cs, int all_proc_freq_cs_size)
{

    multimap<pair<int, int>, pair<string, int>> mmp;

    string val_str = "";
    int clust_num;
    int att_num;
    string att_val;
    int freq;

    int num_sharp = 0;
    int i = 0;

    while (i < all_proc_freq_cs_size)
    {

        if (all_proc_freq_cs[i] == '#' && all_proc_freq_cs[i + 1] == '#')
            i++;
        if (all_proc_freq_cs[i] == '#' && num_sharp < 4)
        { // #1#1#3#2#

            num_sharp++;
            i++;

            while (all_proc_freq_cs[i] != '#')
            {

                val_str = val_str + all_proc_freq_cs[i];
                i++;
            }

            switch (num_sharp)
            {
            case 1:
                clust_num = stoi(val_str);
                break;

            case 2:
                att_num = stoi(val_str);
                break;

            case 3:
                att_val = val_str;
                break;

            case 4:
                freq = stoi(val_str);
                break;
            }
            val_str = "";
        }
        else
        {
            if (i == all_proc_freq_cs_size - 1)
            {
                i++;
            }
            else
            {
                num_sharp = 0;
            }

            mmp.insert({make_pair(clust_num, att_num), make_pair(att_val, freq)});
            val_str = "";
        }

    } // while

    return mmp;
}

string convertToString(char *a)
{
    string s = a;
    return s;
}

tuple<string, int> find_mode_partial_2(const unordered_map<string, int> &histogram)
{

    int mode_count = 0;
    string mode_p;

    for (auto itr = histogram.begin(); itr != histogram.end(); itr++)
    {

        if (itr->second >= mode_count && itr->first != "%")
        { // to ignore the effect of the dumby character "%"
            mode_p = itr->first;
            mode_count = itr->second;
        }
    }

    return make_tuple(mode_p, mode_count); // mode_p;//e.g "3"
}

unordered_map<int, vector<string>> find_mode_global(const int k, const int num_att, multimap<pair<int, int>, pair<string, int>> &mmp, unordered_map<int, vector<int>> &frequencies)
{

    unordered_map<int, vector<string>> global_modes; // clust_num
    unordered_map<string, int> histogram;

    for (size_t clust_num = 0; clust_num < k; ++clust_num)
    {
        for (size_t att_num = 0; att_num < num_att; ++att_num)
        {

            auto it = mmp.equal_range(make_pair(clust_num, att_num)); // it returns multimap<<clust_num , att_num>, <string,int>> and the lower and upper bound of these pairs
            for (auto itr = it.first; itr != it.second; ++itr)
            { // The function returns an iterator of pairs. The pair refers to the bounds. 																									range that includes all the elements in the container which have a key equivalent to k

                histogram[itr->second.first] = histogram[itr->second.first] + itr->second.second; //  itr->second.first = our string(att_val) , itr->second.second = int (att_freq)
            }

            string mode_p;
            int freq;
            tie(mode_p, freq) = find_mode_partial_2(histogram);
            histogram.clear();
            global_modes[clust_num].push_back(mode_p);
            frequencies[clust_num].push_back(freq);

        } // att_num
    }     // clust_num

    return global_modes;
}

void curr_no_points_builder(const int *nums_in_clus_glbl_1, const int num_att, const int k, unordered_map<int, vector<int>> &curr_no_of_points)
{ // clust_num

    for (size_t i = 0; i < k; ++i)
    {
        for (size_t j = 0; j < num_att; ++j)
        {
            curr_no_of_points[i].push_back(nums_in_clus_glbl_1[i]);
        }
    }
}

string vec_to_str(const vector<string> &v)
{
    string result = "";
    for (size_t i = 0; i < v.size(); i++)
    {
        if (i != v.size() - 1)
            result = result + v[i] + ',';
        else
            result = result + v[i];
    }

    return result;
}

vector<string> strToVec(const string mode_unit_str)
{ // mode_unit_str: n,y,y,n,y,y,n,n,n,n,y,n,y,y,n,y

    vector<string> mode_unit_vec;
    string elem = "";
    int i = 0;

    while (i < mode_unit_str.size())
    {

        if (mode_unit_str.at(i) != ',')
        {
            elem = elem + mode_unit_str.at(i);
            i++;
        }
        else
        {
            mode_unit_vec.push_back(elem);
            elem = "";
            i++;
        }
    }

    mode_unit_vec.push_back(elem); // pushing the last elem which doesn't have a chance to get pushed in the while loop

    return mode_unit_vec;
}

string unMapToStr(unordered_map<int, vector<string>> &unMap)
{

    string result_str = ""; // converting the vector of the mode to a string ;

    for (size_t i = 0; i < unMap.size(); i++)
    {
        result_str = result_str + '#' + to_string(i) + '#' + vec_to_str(unMap[i]) + '@';
    }

    return result_str;
}

void cstrToUnmap(const char *global_modes_cs, unordered_map<int, vector<string>> &global_modes)
{

    string clust_num_str = "";
    int clust_num;
    string mode_unit_str = "";

    int i = 0;
    while (i < strlen(global_modes_cs))
    {

        if (global_modes_cs[i] == '#')
        {
            i = i + 1;
            while (global_modes_cs[i] != '#')
            {
                clust_num_str = clust_num_str + global_modes_cs[i];
                i++;
            }
            clust_num = stoi(clust_num_str); // converting clust_num_string to clust_num_integer
            clust_num_str = "";
            i = i + 1; // skipping the second '#' character to get to the data
        }

        while (global_modes_cs[i] != '@')
        {
            mode_unit_str = mode_unit_str + global_modes_cs[i];
            i++;
        }

        // Converting the mode_unit_str to mode_unit_vec
        vector<string> mode_unit_vec = strToVec(mode_unit_str);
        mode_unit_str = "";

        // Now that the clust_num and mode_unit_vec are ready, we store in unordered_map<int,vector<string>> global_modes
        global_modes.insert({clust_num, mode_unit_vec});

        i++; // skipping '@'

    } // while for moving in global_modes_cs
}

float distance_calculate_1(const vector<string> &point, const vector<string> &mode)
{

    int no_of_atts = point.size();
    float dist_t = 0;

    for (int i = 1; i < no_of_atts; i++)
    { // i = 1 -> one-less-att
        if (point.at(i) != mode.at(i - 1))
        { // i-1 -> one-less-att(second one)
            dist_t++;
        }
    }

    return dist_t;
}

vector<vector<vector<string>>> re_cluster(vector<vector<vector<string>>> &points_in_clusters, const unordered_map<int, vector<string>> &global_modes, vector<vector<vector<string>>> &points_to_be_added, unordered_map<int, vector<int>> &frequencies, unordered_map<int, vector<int>> &curr_no_of_points)
{

    vector<vector<vector<string>>> points_to_be_deleted;
    points_to_be_deleted.resize(points_in_clusters.size());

    vector<string> nearest_mode;
    vector<string> curr_mode;
    int target_cluster = 0; // target cluster_number
    float dist = 0.0;
    int num_mode;

    for (size_t i = 0; i < points_in_clusters.size(); ++i)
    { // all cluster_nums
        for (size_t l = 0; l < points_in_clusters.at(i).size(); ++l)
        { // all points in one cluster

            vector<string> point = points_in_clusters.at(i).at(l);
            nearest_mode = global_modes.at(0);
            num_mode = 0;
            target_cluster = 0;

            float min_dist = distance_calculate_2(point, nearest_mode, num_mode, frequencies, curr_no_of_points);

            for (size_t j = 1; j < global_modes.size(); j++)
            {

                curr_mode = global_modes.at(j);
                num_mode = j;
                dist = distance_calculate_2(point, curr_mode, num_mode, frequencies, curr_no_of_points);

                if (dist <= min_dist)
                {
                    nearest_mode = curr_mode;
                    min_dist = dist;
                    target_cluster = j;
                }

            } // most inner-for

            if (i != target_cluster)
            {

                points_to_be_added[target_cluster].push_back(point); // adding the point to its target_cluster

                points_in_clusters[i][l].clear();
                points_in_clusters[i][l].push_back({"ts"});

                points_to_be_deleted[i].push_back(point);

            } // if current cluter is not the same as the target cluster
        }
    }

    return points_to_be_deleted;
}

float E_local_calculator(const vector<vector<vector<string>>> &points_in_clusters, const unordered_map<int, vector<string>> &global_modes)
{

    float E = 0.0;

    for (size_t i = 0; i < points_in_clusters.size(); ++i)
    {
        for (size_t j = 0; j < points_in_clusters.at(i).size(); ++j)
        {
            E = E + distance_calculate_1(points_in_clusters.at(i).at(j), global_modes.at(i));
        }
    }

    return E;
}

string vec_to_str_line(const vector<string> &v)
{
    string result = "";
    for (size_t i = 0; i < v.size(); i++)
    {
        if (i != v.size() - 1)
            result = result + v[i] + ',';
        else
            result = result + v[i] + '\n';
    }

    return result;
}

float distance_calculate_2(const vector<string> &point, const vector<string> &mode, const int num_mode, unordered_map<int, vector<int>> &frequencies,
                           unordered_map<int, vector<int>> &curr_no_of_points)
{

    int num_att = point.size(); //- 1;// one-less-att
    float dist_t = 0.0;

    for (int i = 1; i < num_att; i++)
    { // i = 1 -> one-less-att

        if (point.at(i) != mode.at(i - 1))
        { // i-1 -> one-less-att(second one)
            dist_t++;
        }
        else
        { // if they are equal , we see how much equal they are

            int freq_of_att = frequencies.at(num_mode).at(i - 1);
            int current_no_of_points = curr_no_of_points.at(num_mode).at(i - 1);

            dist_t = dist_t + (1.0 - ((float)freq_of_att / (float)current_no_of_points));

        } // else

    } // for

    return dist_t;
}

vector<vector<vector<string>>> all_points_in_clusters_extractor(char *result_cs, int result_cs_size, int k)
{
    vector<vector<vector<string>>> all_points_in_clusters;
    all_points_in_clusters.resize(k);
    vector<string> point;
    string clust_num = "";
    string att_val = "";

    int i = 0;
    while (i < result_cs_size)
    {

        if (result_cs[i] == '@')
        {
            clust_num = "";
            i++;
            while (result_cs[i] != '@')
            {
                clust_num = clust_num + result_cs[i];
                i++;
            }
        } // if

        i++;
        while (result_cs[i] != '@' && i < result_cs_size)
        {

            while (result_cs[i] != ',' && result_cs[i] != '#')
            {

                att_val = att_val + result_cs[i];
                i++;
            }

            point.push_back(att_val);
            att_val = "";

            if (result_cs[i] == '#')
            {
                all_points_in_clusters[stoi(clust_num)].push_back(point);
                point.clear();
            }
            i++;
        }
    }
    return all_points_in_clusters;
}

//
template <class T>
ostream &operator<<(ostream &os, const vector<T> &v)
{
    for (size_t i = 0; i < v.size(); i++)
    {
        os << v[i] << " ";
    }
    os << endl;
    return os;
}

unordered_map<int, vector<string>> points_to_attribute0(vector<vector<vector<string>>> &points_in_clusters)
{

    unordered_map<int, vector<string>> attribute0; // int-> clust_num = 0 , string of attribute 0

    for (size_t clus_num = 0; clus_num < points_in_clusters.size(); clus_num++)
    { // iterate through cluster numbers
        for (size_t i = 0; i < points_in_clusters[clus_num].size(); ++i)
        { // iterate through points in a cluster

            vector<string> point = points_in_clusters.at(clus_num).at(i);
            attribute0[clus_num].push_back(point[0]); //
        }
    }

    return attribute0;
}

unordered_map<int, unordered_map<string, int>> freq_table0_builder(unordered_map<int, vector<string>> &attribute0)
{

    unordered_map<int, unordered_map<string, int>> freq_table; // int -> clust_num

    for (size_t j = 0; j < attribute0.size(); j++)
    { // clust_num

        freq_table[j] = histogram_of_one_att_builder(attribute0.at(j));
    }

    return freq_table;
}

void pre_rec_f1_acc_v2(int ext_itr, int num_points, vector<vector<vector<string>>> &points_in_clusters, vector<float> &pre_w, vector<float> &rec_w, vector<float> &f1_w,
                       vector<float> &pre_M, vector<float> &rec_M, vector<float> &f1_M, vector<float> &acc)
{

    int k = points_in_clusters.size();
    unordered_map<int, vector<string>> attributes0 = points_to_attribute0(points_in_clusters);
    unordered_map<int, unordered_map<string, int>> freq_table0 = freq_table0_builder(attributes0); // int->clust_num
    unordered_map<int, vector<string>> class_lables;                                               // int->clust_num , vector of all max values to be able to choose from it later

    vector<string> covered_labels(k, "");
    vector<int> covered_clusters;
    for (size_t i = 0; i < k; ++i)
    { // number of class labels

        string label_max;
        int label_max_num = 0;
        int clust_num;
        bool check = false;

        for (size_t j = 0; j < freq_table0.size(); ++j)
        { // to iterate through freq_table0
            for (auto itr = freq_table0.at(j).begin(); itr != freq_table0.at(j).end(); ++itr)
            {

                // this helps to find the max, second max , third max and etc.
                if (find(covered_clusters.begin(), covered_clusters.end(), j) == covered_clusters.end())
                {
                    if (find(covered_labels.begin(), covered_labels.end(), itr->first) == covered_labels.end())
                    { // if not available in covered_labels

                        if (itr->second > label_max_num)
                        { // i.e. and it's bigger than label_max_num
                            label_max_num = itr->second;
                            label_max = itr->first;
                            clust_num = j;
                            check = true;
                        }
                    }
                }
            }
        } // to iterate through freq_table0

        if (check == true)
        {

            covered_clusters.push_back(clust_num);
            covered_labels.push_back(label_max);
            class_lables[clust_num].push_back(label_max);
        }
        else
        { // To cover this case when we are missing a class label with no member in the remaining cluster.
          //  So  we have to assign the remaining class label and cluster number in the special case
            cout << "FALSE" << endl;

            // To find all the exising class labels
            vector<string> all_labels;
            for (size_t j = 0; j < freq_table0.size(); ++j)
            { // to iterate through freq_table0
                for (auto itr = freq_table0.at(j).begin(); itr != freq_table0.at(j).end(); ++itr)
                {

                    if (find(all_labels.begin(), all_labels.end(), itr->first) == all_labels.end())
                    { // if Not found
                        all_labels.push_back(itr->first);
                    }
                }
            } // for to iterate through freq_table0

            // to identify which cluster has no label
            int missingCluster;
            string missingClusterLabel;
            for (size_t h = 0; h < k; ++h)
            {
                if (find(covered_clusters.begin(), covered_clusters.end(), h) == covered_clusters.end())
                {
                    missingCluster = h;
                }
            }

            // To identify which label has no cluster
            for (size_t i = 0; i < k; ++i)
            {
                if (find(covered_labels.begin(), covered_labels.end(), all_labels.at(i)) == covered_labels.end())
                {
                    missingClusterLabel = all_labels.at(i);
                }
            }

            // to match these two together
            class_lables[missingCluster].push_back(missingClusterLabel);
            covered_clusters.push_back(missingCluster);
            covered_labels.push_back(missingClusterLabel);

        } // else
    }

    unordered_map<int, float> precision; // num_clust , precision of the cluster
    unordered_map<int, float> recall;
    unordered_map<int, float> f1score;

    float macro_precision = 0.0;
    float macro_recall = 0.0;
    float macro_f1score = 0.0;

    float wei_precision = 0.0;
    float wei_recall = 0.0;
    float wei_f1score = 0.0;

    float accuracy = 0.0;

    // TP,FP
    int sumOfTps = 0;
    for (size_t clust_num = 0; clust_num < freq_table0.size(); ++clust_num)
    {

        int fp_num = 0;

        string tp = class_lables.at(clust_num).at(0);
        unordered_map<string, int> m = freq_table0.at(clust_num);
        int tp_num = 0;
        if (m.count(tp)) // If class labels are entered manually then maybe we end up with no tp or tp = 0
            tp_num = freq_table0.at(clust_num).at(tp);

        sumOfTps = sumOfTps + tp_num;

        for (auto itr = freq_table0.at(clust_num).begin(); itr != freq_table0.at(clust_num).end(); ++itr)
        {
            fp_num = fp_num + itr->second;
        }
        fp_num = fp_num - tp_num;
        precision[clust_num] = (float)tp_num / (tp_num + fp_num); // pecision of cluster

        int fn_num = 0;
        for (size_t i = 0; i < k; ++i)
        {
            if (i != clust_num)
            {
                unordered_map<string, int> m = freq_table0.at(i);
                if (m.count(tp))
                { // To check if there's FN in other classes or not
                    fn_num = fn_num + freq_table0.at(i).at(tp);
                }
            }
        }
        recall[clust_num] = (float)tp_num / (tp_num + fn_num);

        if (precision.at(clust_num) == 0 || recall.at(clust_num) == 0)
            f1score[clust_num] = 0.0;
        else
            f1score[clust_num] = (float)2 * precision[clust_num] * recall[clust_num] / (precision[clust_num] + recall[clust_num]);
    }

    // Macro
    for (int i = 0; i < k; ++i)
    {

        macro_precision = macro_precision + precision.at(i);
        macro_recall = macro_recall + recall.at(i);
        macro_f1score = macro_f1score + f1score.at(i);
    }
    macro_precision = (float)macro_precision / k;
    macro_recall = (float)macro_recall / k;
    macro_f1score = (float)macro_f1score / k;
    accuracy = (float)sumOfTps / num_points;
    cout << "*******Ext-itr: " << ext_itr << "*********" << endl;
    cout << "Accuracy = " << accuracy << endl;
    cout << "Macro precision = " << macro_precision << endl;
    cout << "Macro recall = " << macro_recall << endl;
    cout << "Macro f1score =" << macro_f1score << endl;
    cout << "*******Ext-itr: " << ext_itr << "*********" << endl;

    // Weighted
    for (size_t clust_num = 0; clust_num < freq_table0.size(); ++clust_num)
    {
        string tp = class_lables.at(clust_num).at(0);
        int tp_num = 0;
        unordered_map<string, int> m = freq_table0.at(clust_num);
        if (m.count(tp)) // If class labels are entered manually then maybe we end up with no tp or tp = 0
            tp_num = freq_table0.at(clust_num).at(tp);

        wei_precision = wei_precision + tp_num * precision.at(clust_num);
        wei_recall = wei_recall + tp_num * recall.at(clust_num);
        wei_f1score = wei_f1score + tp_num * f1score.at(clust_num);
    }
    wei_precision = wei_precision / num_points;
    wei_recall = wei_recall / num_points;
    wei_f1score = wei_f1score / num_points;

    pre_M.push_back(macro_precision);
    rec_M.push_back(macro_recall);
    f1_M.push_back(macro_f1score);

    pre_w.push_back(wei_precision);
    rec_w.push_back(wei_recall);
    f1_w.push_back(wei_f1score);

    acc.push_back(accuracy);
}

void rd_read_initial_clusters(const string fileName, const int *start_line, const int *end_line, const int k, const int rank, vector<vector<vector<string>>> &points_in_clusters)
{

    string line;
    ifstream myfile(fileName, ios::in);

    vector<string> point;
    vector<vector<string>> point_s_lc;

    if (!myfile.is_open())
    {
        cout << "Failed to open" << endl;
        exit(0);
    }

    int my_start_line = start_line[rank];
    int my_end_line = end_line[rank];
    int numPointsLocal = my_end_line - my_start_line;

    // Defining the size of inner clusters within each process
    // e.g: 10 points - 3 clusters => 3 , 3 , 4
    // e.g: 2 points - 3 clusters => 0 , 0 , 2 RARE CASE
    int inner_cluster_len = numPointsLocal / k;
    int inner_cluster_rm = numPointsLocal % k;
    int inner_cluster_size[k] = {};

    for (int i = 0; i < k - 1; ++i)
    {
        inner_cluster_size[i] = inner_cluster_len;
    }
    inner_cluster_size[k - 1] = inner_cluster_len + inner_cluster_rm;

    /////////////////////////////////////////////////////////////

    int myShare = 0;
    int line_counter = 0;

    //***************************reading point_s_lc
    while (getline(myfile, line))
    {

        if (myShare <= numPointsLocal) // controling the # of all the points in 1 process
        {
            if (my_start_line <= line_counter && line_counter < my_end_line)
            {
                myShare++;

                stringstream sline(line);
                string word;
                while (getline(sline, word, ','))
                {
                    point.push_back(word);
                }

                point_s_lc.push_back(point);

                point.clear();
                line_counter++;
            }
            else
            {

                line_counter++;
            }
        }
        else
        {

            myfile.close();
        }

    } // While

    //*************************randomizing locally:
    vector<vector<string>> rdm_point_s_lc;

    while (!point_s_lc.empty())
    {

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        std::uniform_int_distribution<int> distribution(0, point_s_lc.size() - 1);

        vector<string> obj;
        int index = distribution(generator);
        obj = point_s_lc.at(index);
        rdm_point_s_lc.push_back(obj);
        point_s_lc.erase(point_s_lc.begin() + index);
        obj.clear();
    }

    //*************************reading initial clusters:
    int num_cluster = 0;
    int cnt = 0;

    point.clear();

    while (!rdm_point_s_lc.empty())
    {

        if (cnt < inner_cluster_size[num_cluster]) // arranging the points in inner clusters based on their sizes
        {

            point = rdm_point_s_lc.at(0); // we regularly deduct from the beginning of the vector
            rdm_point_s_lc.erase(rdm_point_s_lc.begin() + 0);
            points_in_clusters[num_cluster].push_back(point);
            cnt++;
            point.clear();

        } // new if

        else
        {
            cnt = 0;
            num_cluster++;

            point = rdm_point_s_lc.at(0);
            rdm_point_s_lc.erase(rdm_point_s_lc.begin() + 0);

            points_in_clusters[num_cluster].push_back(point);
            cnt++;
            point.clear();
        }

    } // While
}

template <class T>
T average_calculater(vector<T> &v)
{
    T ave = 0;
    for (size_t i = 0; i < v.size(); i++)
    {
        ave = ave + v.at(i);
    }
    ave = (T)ave / v.size();
    return ave;
}

void read_initial_clusters_nonRD(const string fileName, const int *start_line, const int *end_line, const int k, const int rank, vector<vector<vector<string>>> &points_in_clusters)
{

    string line;
    ifstream myfile(fileName, ios::in);

    vector<string> point;

    if (!myfile.is_open())
    {
        cout << "Failed to open" << endl;
        exit(0);
    }

    int my_start_line = start_line[rank];
    int my_end_line = end_line[rank];
    int numPointsLocal = my_end_line - my_start_line;

    /////////////////////////////////////////////////////////////

    int num_cluster;
    int cnt = 0;
    int myShare = 0;
    int line_counter = 0;

    while (getline(myfile, line))
    {

        if (myShare <= numPointsLocal) // controling the # of all the points in 1 process
        {
            if (my_start_line <= line_counter && line_counter < my_end_line)
            {
                myShare++;

                stringstream sline(line);
                string word;

                while (getline(sline, word, ','))
                {
                    point.push_back(word);
                }
                num_cluster = stoi(point[0]); // this is the indicator of initial clusters for each point
                point.erase(point.begin());   // then we remove it to end up with a clear-cut point
                points_in_clusters[num_cluster].push_back(point);
                cnt++;
                point.clear();
                line_counter++;
            }
            else
            {

                line_counter++;
            }
        }
        else
        {

            myfile.close();
        }

    } // While
}

unordered_map<int, vector<string>> points_to_attributes_nonClust(vector<vector<string>> &rdm_point_s_lc, int k)
{

    unordered_map<int, vector<string>> attributes_nonClust; // int-> attribute_num , for example, all the middle attributes

    for (size_t i = 0; i < rdm_point_s_lc.size(); ++i)
    { // iterate through points in a cluster

        vector<string> point = rdm_point_s_lc.at(i);
        for (size_t i = 1; i < point.size(); i++)
        { // i = 1 -> one-less-att

            attributes_nonClust[i - 1].push_back(point[i]); // i-1 -> one-less-att (the first one)
        }
    }

    return attributes_nonClust;
}

unordered_map<int, unordered_map<string, int>> one_proc_freq_builder_nonClust(unordered_map<int, vector<string>> &attributes_nonClust)
{

    // unordered_map<int , unordered_map<string , int>> histogram_of_all_atts;//int -> att_num
    unordered_map<int, unordered_map<string, int>> one_proc_freq_nonClust; // int -> att_num

    for (size_t i = 0; i < attributes_nonClust.size(); i++)
    { // att_num

        one_proc_freq_nonClust[i] = histogram_of_one_att_builder(attributes_nonClust.at(i));
    }

    return one_proc_freq_nonClust;
}

string unMap_3d_ToStr_nonClust(unordered_map<int, unordered_map<string, int>> &unMap)
{ // a function to convert the one_proc_freq to string :
  //  #att_num#att_val#frequency##att_num#att_val#frequency.....
    string result_str = "";

    for (size_t att_num = 0; att_num < unMap.size(); att_num++)
    {

        unordered_map<string, int> temp = unMap[att_num];
        string att_val;
        int freq;

        for (auto itr = temp.begin(); itr != temp.end(); itr++)
        {
            att_val = itr->first;
            freq = itr->second;
            result_str = result_str + '#' + to_string(att_num) + '#' + att_val + '#' + to_string(freq);
        }
    }
    return result_str; // e.g #0#0#2#1#0#0#1#5#0#1#2#2#0#1#3#2#0#1#1#2#1#0#3#3#1#0#2#3#1#1#2#1#1#1#1#3#1#1#3#2
}

multimap<int, pair<string, int>> mpp_builder(const char *all_proc_freq_nonClust_cs, int all_proc_freq_nonClust_cs_size, const int rank)
{ // multimap, because we are going to have, for example, att_num = 0 for n number of processes

    multimap<int, pair<string, int>> mpp;

    string val_str = "";
    // int clust_num;
    int att_num;
    string att_val;
    int freq;

    int num_sharp = 0;
    int i = 0;

    while (i < all_proc_freq_nonClust_cs_size)
    {

        if (all_proc_freq_nonClust_cs[i] == '#' && all_proc_freq_nonClust_cs[i + 1] == '#') // To ignore the end of a string of att_num,att_val,freq
            i++;
        if (all_proc_freq_nonClust_cs[i] == '#' && num_sharp < 3)
        { // #1#1#3#

            num_sharp++;
            i++;

            while (all_proc_freq_nonClust_cs[i] != '#')
            {

                val_str = val_str + all_proc_freq_nonClust_cs[i];
                i++;
            }

            switch (num_sharp)
            {

            case 1:
                att_num = stoi(val_str);
                break;

            case 2:
                att_val = val_str;
                break;

            case 3:
                freq = stoi(val_str);
                break;
            }
            val_str = "";
        }
        else
        {
            if (i == all_proc_freq_nonClust_cs_size - 1)
            {
                i++;
            }
            else
            {
                num_sharp = 0;
            }

            mpp.insert({att_num, make_pair(att_val, freq)});
            val_str = "";
        }

    } // while

    return mpp;
}

unordered_map<int, unordered_map<string, int>> freq_table_global_builder(const int num_att, multimap<int, pair<string, int>> &mpp)
{

    unordered_map<int, unordered_map<string, int>> freq_table_global; // att_num
    unordered_map<string, int> histogram;

    for (size_t att_num = 0; att_num < num_att; ++att_num)
    {

        auto it = mpp.equal_range(att_num); // it returns multimap<<clust_num , att_num>, <string,int>> and the lower and upper bound of these pairs
        for (auto itr = it.first; itr != it.second; ++itr)
        { // The function returns an iterator of pairs. The pair refers to the bounds  																									range that includes all the elements in the container which have a key equivalent to k

            histogram[itr->second.first] = histogram[itr->second.first] + itr->second.second; //  itr->second.first = our string(att_val) , itr->second.second = int (att_freq)
        }

        freq_table_global[att_num] = histogram;
        histogram.clear();

    } // att_num

    return freq_table_global;
}

multimap<float, vector<string>, greater<float>> density_builder_dmm(vector<vector<string>> &rdm_point_s_lc, unordered_map<int, unordered_map<string, int>> &freq_table_global)
{ //, float& ave_dens

    multimap<float, vector<string>, greater<float>> density;

    for (size_t ptn_num = 0; ptn_num < rdm_point_s_lc.size(); ++ptn_num)
    {

        vector<string> point = rdm_point_s_lc.at(ptn_num);
        float dens = 0.0;
        for (size_t j = 0; j < freq_table_global.size(); ++j)
        { // one-less att

            string att_val = point[j + 1];
            unordered_map<string, int> freq_table_global_of_att_j = freq_table_global.at(j);
            dens = dens + freq_table_global_of_att_j.at(att_val);
        }

        dens = (float)dens / freq_table_global.size(); // divided by num_att
        density.insert(make_pair(dens, point));
    }

    // ave_dens = ave_dens/point_s.size();
    return density;
}

vector<vector<vector<string>>> re_cluster(int k, const unordered_map<int, vector<string>> &global_modes,
                                          unordered_map<int, vector<int>> &frequencies, unordered_map<int, vector<int>> &curr_no_of_points, vector<vector<float>> &k_distXdens,
                                          const multimap<float, vector<string>, greater<float>> &cpyOfdens_point, const int rank)
{

    vector<vector<vector<string>>> new_points_in_clusters;
    new_points_in_clusters.resize(k);

    vector<string> nearest_mode;
    vector<string> curr_mode;
    int target_cluster = 0; // target cluster_number
    float dist = 0.0;
    int num_mode;
    int num_points_cntr = 0;

    for (auto itr = cpyOfdens_point.begin(); itr != cpyOfdens_point.end(); ++itr)
    { // using cpyOfdens_point instead of point_in_clusters to maintain the order

        vector<string> point = itr->second;
        nearest_mode = global_modes.at(0);
        num_mode = 0;
        target_cluster = 0;

        float min_dist = distance_calculate_2(point, nearest_mode, num_mode, frequencies, curr_no_of_points);

        float dist0Xdens = min_dist * itr->first;           // itr->first is the density of the point we are examining.
        k_distXdens[num_points_cntr].push_back(dist0Xdens); // We have "n" rows(1 for each point) & each row of k_distXdens has "k" elements.(Dist from each mode*dens of the point).

        for (size_t j = 1; j < global_modes.size(); j++)
        {

            curr_mode = global_modes.at(j);
            num_mode = j;
            dist = distance_calculate_2(point, curr_mode, num_mode, frequencies, curr_no_of_points);
            k_distXdens[num_points_cntr].push_back(dist * itr->first); //

            if (dist <= min_dist)
            {
                nearest_mode = curr_mode;
                min_dist = dist;
                target_cluster = j;
            }

        } // inner-for
        num_points_cntr++;
        new_points_in_clusters[target_cluster].push_back(point);
    }

    return new_points_in_clusters;
}

vector<string> EmpClustHandler(const int k, vector<vector<float>> &k_distXdens,
                               const multimap<float, vector<string>, greater<float>> cpyOfdens_point, const int emptyClusterIndex, float &myLocalMax, int &myMaxDistXDensIndex, float *all_maxes, const int rank, const int numtasks, vector<float> &min_distXdens)
{

    vector<string> modeForEmp;
    int num_points = cpyOfdens_point.size();
    int emptyClustIndex;

    int cnt = 0;

    /*We start iterating over k_distXdens and get the min for each point and save it in min_distXdens*/
    for (int j = 0; j < num_points; ++j)
    {

        vector<float> oneRow; // k numbers for each point.
        for (int num_clust = 0; num_clust < k; ++num_clust)
        {
            if (num_clust != emptyClusterIndex)
            { // every mode except the empty cluster's mode because we are tryig to find a centroid for this cluster with the help of other clusters' mode
                oneRow.push_back(k_distXdens.at(j).at(num_clust));
            }
        }
        float minOfEachRow = *min_element(oneRow.begin(), oneRow.end());
        min_distXdens.push_back(minOfEachRow);

    } // for every point

    myMaxDistXDensIndex = std::max_element(min_distXdens.begin(), min_distXdens.end()) - min_distXdens.begin(); // Finding the point with MAX densXdist IN EACH PROCESS
    myLocalMax = *max_element(min_distXdens.begin(), min_distXdens.end());

    MPI_Allgather(&myLocalMax, 1, MPI_FLOAT, all_maxes, 1, MPI_FLOAT, MPI_COMM_WORLD);

    const auto itr1 = cpyOfdens_point.begin();
    const auto itr2 = next(itr1, myMaxDistXDensIndex); // advances itr1 with the amount of maxDistDensIndex and returns an itrator named itr2
    vector<string> cand_modeForEmp = itr2->second;     // every process prepares a candidate based on their maximum
    cand_modeForEmp.erase(cand_modeForEmp.begin());    // Deleting the expert column
    string cand_modeForEmp_str = vec_to_str(cand_modeForEmp);

    char *cand_modeForEmp_cs = new char[cand_modeForEmp_str.size() + 1]();
    strcpy(cand_modeForEmp_cs, cand_modeForEmp_str.c_str());

    // Now every process has all the maximums--->identifying the process with the max distXdens point from other modes that their clusters aren't empty.
    int global_max_rank = 0;
    float global_max = all_maxes[global_max_rank];

    for (size_t i = 1; i < numtasks; ++i)
    {
        if (all_maxes[i] > global_max)
        {
            global_max = all_maxes[i];
            global_max_rank = i;
        }
    }

    // modeForEmp is sent to all other processes by the process that has it.
    int modeForEmp_cs_size = 0;
    if (rank == global_max_rank)
    {
        modeForEmp_cs_size = cand_modeForEmp_str.size();
    }
    MPI_Bcast(&modeForEmp_cs_size, 1, MPI_INT, global_max_rank, MPI_COMM_WORLD);
    char *modeForEmp_cs = new char[modeForEmp_cs_size + 1]();

    if (rank == global_max_rank)
    {
        strcpy(modeForEmp_cs, cand_modeForEmp_cs);
    }
    MPI_Bcast(modeForEmp_cs, modeForEmp_cs_size, MPI_CHAR, global_max_rank, MPI_COMM_WORLD);

    string modeForEmp_str = modeForEmp_cs;
    modeForEmp = strToVec(modeForEmp_cs);

    return modeForEmp;
}

float distance_calculate_1_v2(const vector<string> &point1, const vector<string> &point2)
{ // Btw 2 points

    int no_of_atts = point1.size();
    float dist_t = 0;

    for (int i = 1; i < no_of_atts; i++)
    { // i = 1 -> one-less-att
        if (point1.at(i) != point2.at(i))
        { // i-> not one-less-att(second one) because we are comparing two actual point not a point and a mode
            dist_t++;
        }
    }

    return dist_t;
}

unordered_map<int, vector<string>> find_mode_global_partial(const int k, const int num_att, multimap<pair<int, int>, pair<string, int>> &mmp)
{ // modified to produce what was needed for partial global mode

    unordered_map<int, vector<string>> global_modes_partial; // clust_num
    unordered_map<string, int> histogram;

    for (size_t clust_num = 0; clust_num < k; ++clust_num)
    {
        for (size_t att_num = 0; att_num < num_att; ++att_num)
        {

            auto it = mmp.equal_range(make_pair(clust_num, att_num)); // it returns multimap<<clust_num , att_num>, <string,int>> and the lower and upper bound of these pairs
            for (auto itr = it.first; itr != it.second; ++itr)
            { // The function returns an iterator of pairs. The pair refers to the bounds.

                histogram[itr->second.first] = histogram[itr->second.first] + itr->second.second; //  itr->second.first = our string(att_val) , itr->second.second = int (att_freq)
            }

            string mode_p;
            int freq;
            tie(mode_p, freq) = find_mode_partial_2(histogram);
            histogram.clear();
            global_modes_partial[clust_num].push_back(mode_p);

        } // att_num
    }     // clust_num

    return global_modes_partial;
}

void rd_read_initial_clusters(const string fileName, const int *start_line, const int *end_line, const int rank, vector<vector<string>> &rdm_point_s_lc)
{

    string line;
    ifstream myfile(fileName, ios::in);

    vector<string> point;
    vector<vector<string>> point_s_lc;

    if (!myfile.is_open())
    {
        cout << "Failed to open" << endl;
        exit(0);
    }

    int my_start_line = start_line[rank];
    int my_end_line = end_line[rank];
    int numPointsLocal = my_end_line - my_start_line;

    /////////////////////////////////////////////////////////////

    int myShare = 0;
    int line_counter = 0;

    //***************************reading point_s_lc
    while (getline(myfile, line))
    {

        if (myShare <= numPointsLocal) // controling the # of all the points in 1 process
        {
            if (my_start_line <= line_counter && line_counter < my_end_line)
            {
                myShare++;

                stringstream sline(line);
                string word;
                while (getline(sline, word, ','))
                {
                    point.push_back(word);
                }

                point_s_lc.push_back(point);

                point.clear();
                line_counter++;
            }
            else
            {

                line_counter++;
            }
        }
        else
        {

            myfile.close();
        }

    } // While

    //*************************randomizing locally:

    while (!point_s_lc.empty())
    {

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        std::uniform_int_distribution<int> distribution(0, point_s_lc.size() - 1);

        vector<string> obj;
        int index = distribution(generator);
        obj = point_s_lc.at(index);
        rdm_point_s_lc.push_back(obj);
        point_s_lc.erase(point_s_lc.begin() + index);
        obj.clear();
    }
}

map<int, vector<string>> points_to_attributes_map(vector<vector<string>> &point_s)
{

    map<int, vector<string>> attributes; // int-> attribute_num , for example, all the middle attributes

    for (size_t i = 0; i < point_s.size(); ++i)
    { // iterate through points in a cluster

        vector<string> point = point_s.at(i);
        for (size_t j = 1; j < point.size(); j++)
        {                                          // j = 1 -> one-less-att
            attributes[j - 1].push_back(point[j]); // j-1 -> one-less-att (the first one)
        }
    }

    return attributes;
}

map<string, int> histogram_of_one_att_builder_map(const vector<string> &v)
{

    map<string, int> histogram_of_one_att;

    for (auto x : v)
    {
        histogram_of_one_att[x]++;
    }

    return histogram_of_one_att; // mode_p;//e.g "3"
}

map<int, map<string, int>> freq_table_builder_map(map<int, vector<string>> &attributes)
{

    map<int, map<string, int>> freq_table; // int -> att_num

    for (size_t j = 0; j < attributes.size(); j++)
    { // att_num

        freq_table[j] = histogram_of_one_att_builder_map(attributes.at(j));
    }

    return freq_table;
}

vector<pair<string, string>> pairsOfattVals_builder(map<int, map<string, int>> &freq_table_map, int att_num)
{

    vector<pair<string, string>> pairs;
    map<string, int> m = freq_table_map.at(att_num);

    for (auto itr1 = m.begin(); itr1 != m.end(); ++itr1)
    {
        string x = itr1->first;
        auto fromItr = next(itr1);
        for (auto itr2 = fromItr; itr2 != m.end(); ++itr2)
        {
            pairs.push_back(make_pair(x, itr2->first));
        }
    }

    return pairs;
}

vector<string> att_vals_builder(map<int, map<string, int>> &freq_table_map, int att_num)
{

    vector<string> att_vals;
    map<string, int> freq_table_i = freq_table_map.at(att_num);

    for (auto itr = freq_table_i.begin(); itr != freq_table_i.end(); ++itr)
    {
        att_vals.push_back(itr->first);
    }

    return att_vals;
}

string my_local_att_vals_str_builder(map<int, map<string, int>> &freq_table_map, const int num_att)
{

    unordered_map<int, vector<string>> my_local_att_vals;

    for (size_t i = 0; i < num_att; ++i)
    {

        vector<string> att_vals = att_vals_builder(freq_table_map, i);
        my_local_att_vals[i] = att_vals;
    }

    string lc_att_vals_str = "";
    for (size_t i = 0; i < num_att; ++i)
    {

        lc_att_vals_str = lc_att_vals_str + '@' + to_string(i) + '@';
        vector<string> v = my_local_att_vals.at(i);

        for (size_t j = 0; j < v.size(); ++j)
        {
            lc_att_vals_str = lc_att_vals_str + v.at(j) + '#';
        }
    }

    return lc_att_vals_str;
}

map<int, vector<string>> gl_att_vals_builder(const char *glb_att_vals_cs, const int glb_att_vals_cs_size)
{

    map<int, vector<string>> glbl_att_vals;

    int i = 0;
    int att_num = 0;
    string att_number = "";
    string att_val = "";

    while (i < glb_att_vals_cs_size)
    {

        if (glb_att_vals_cs[i] == '@')
        {

            att_number = "";
            i++;

            while (glb_att_vals_cs[i] != '@')
            {

                att_number = att_number + glb_att_vals_cs[i];
                i++;
            }

            att_num = stoi(att_number);
            i++;

        } // if @

        while (glb_att_vals_cs[i] != '#')
        {

            att_val = att_val + glb_att_vals_cs[i];
            i++;
        }

        glbl_att_vals[att_num].push_back(att_val);
        att_val = "";
        i++;

    } // while

    return glbl_att_vals;
}

void fourNums_4eachAttVal_builder(pair<string, string> &myPair, int att_num_pri, map<int, vector<string>> &glbl_unique_att_vals, int att_num_sec, vector<vector<string>> &point_s,
                                  map<int, map<pair<string, string>, map<int, map<string, vector<int>>>>> &partial_table_one_proc, string &partial_table_str)
{

    vector<int> fourNums;
    string first_elem = myPair.first;
    string sec_elem = myPair.second;
    vector<string> att_vals = glbl_unique_att_vals.at(att_num_sec);

    for (size_t i = 0; i < att_vals.size(); ++i)
    { // for -->to find the max amount

        string att_val = att_vals.at(i);
        int p_fst = 0;
        int p_not_sec = 0;
        int cnt_fst_elem = 0;
        int cnt_sec_elem = 0;

        for (size_t j = 0; j < point_s.size(); ++j)
        { // to calculate the probability of

            vector<string> point = point_s.at(j);

            if (point.at(att_num_pri + 1) == first_elem)
            {
                cnt_fst_elem++;
                if (point.at(att_num_sec + 1) == att_val)
                {
                    p_fst++;
                }
            }

            if (point.at(att_num_pri + 1) == sec_elem)
            {
                cnt_sec_elem++;
                if (point.at(att_num_sec + 1) != att_val)
                {
                    p_not_sec++;
                }
            }

        } // point_s

        fourNums.push_back(p_fst);
        fourNums.push_back(p_not_sec);
        fourNums.push_back(cnt_fst_elem);
        fourNums.push_back(cnt_sec_elem);
        partial_table_one_proc[att_num_pri][myPair][att_num_sec][att_val] = fourNums;
        partial_table_str = partial_table_str + '@' + to_string(att_num_pri) + '@' + first_elem + ',' + sec_elem + '%' + to_string(att_num_sec) + '$' + att_val + '*' + to_string(p_fst) + '#' + to_string(p_not_sec) + '#' + to_string(cnt_fst_elem) + '#' + to_string(cnt_sec_elem) + '#';

        fourNums.clear();

    } // att_vals
    //        cout<<"for pair : "<<first_elem<<" , "<<sec_elem<<"\t"<<"max_prob regarding att_num_"<<att_num_sec<<" is : "<<max_prob<<endl;
}

float dist_calculator_sig_pp(vector<string> point1, vector<string> point2, map<int, map<pair<string, string>, float>> &dist_pair)
{

    int num_att = point1.size() - 1;
    float dist_t = 0;

    for (size_t i = 0; i < num_att; ++i)
    {
        dist_t = dist_t + dist_pair_finder_pp(dist_pair, point1, point2, i);
    }

    return dist_t;
}

float dist_pair_finder_pp(map<int, map<pair<string, string>, float>> &dist_pair, vector<string> &point1, vector<string> &point2, int att_num)
{

    map<pair<string, string>, float> pairsOfOneAtt = dist_pair.at(att_num);
    float distBtwPair = 0;

    for (auto itr = pairsOfOneAtt.begin(); itr != pairsOfOneAtt.end(); ++itr)
    {
        string x = itr->first.first;
        string y = itr->first.second;
        if ((x == point1.at(att_num + 1) && y == point2.at(att_num + 1)) || (x == point2.at(att_num + 1) && y == point1.at(att_num + 1)))
        {
            distBtwPair = itr->second;
        }
    }

    return distBtwPair;
}

bool operator==(const myKey &o1, const myKey &o)
{
    return o1.pri == o.pri && o1.myPair.first == o.myPair.second && o1.myPair.second == o.myPair.second && o1.sec == o.sec && o1.att_val == o.att_val;
}

bool operator<(const myKey &o1, const myKey &o)
{
    return (o1.pri < o.pri) ||
           ((o1.pri == o.pri) && (o1.myPair.first < o.myPair.first)) ||
           ((o1.pri == o.pri) && (o1.myPair.first == o.myPair.first) && (o1.myPair.second < o.myPair.second)) ||
           ((o1.pri == o.pri) && (o1.myPair.first == o.myPair.first) && (o1.myPair.second == o.myPair.second) && (o1.sec < o.sec)) ||
           ((o1.pri == o.pri) && (o1.myPair.first == o.myPair.first) && (o1.myPair.second == o.myPair.second) && (o1.sec == o.sec) && (o1.att_val < o.att_val));
}