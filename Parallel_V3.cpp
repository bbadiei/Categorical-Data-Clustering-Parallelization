// Final modified V3
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
void read_initial_clusters(const string fileName, const int *start_line, const int *end_line, const int k, const int rank, vector<vector<vector<string>>> &points_in_clusters);
ostream &operator<<(std ::ostream &os, const vector<string> &v);
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
void curr_no_points_builder(const int num_att, const vector<vector<vector<string>>> &points_in_clusters, unordered_map<int, vector<int>> &curr_no_of_points);
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

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);
    int rank, numtasks;
    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    char fileName[] = "BreastCancerDataSet.csv";
    int k = 2; // we need to clarify # of clusters in this version of parallelelized KMODE, because it's independant of # of processes.
               // CarDataSet.csv , BreastCancerDataSet , BCToy , CongressionalVotesDataSet , MushroomDataSet
               // Mushroom_diff_order  ,  CongressionalVotes_diff_order  , BreastCancer_diff_order , vote_toy , vote_toy_20
               // zoo.csv , audiology.standardized , balance-scale , letter-recognition , MushroomDataSet_512 , soybean-small

    /****line_att_counter*****/
    int num_att = 0;
    int numPointsTotal = line_att_counter(fileName, num_att);
    if (rank == Master)
    {
        cout << "numPointsTotal: " << numPointsTotal << endl;
        cout << "num_att : " << num_att << endl;
        cout << " K : " << k << endl;
    }

    /*****read_initial_clusters*****/
    /*****Defining start_line and end_line for each process of MPI (numPointsLocal also implicitly*****/
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
    vector<float> E_all;

    int total_iterations = 100; // repeating the whole process of clustering 100 times, since we start from random initial seeds.
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
        rd_read_initial_clusters(fileName, start_line, end_line, k, rank, points_in_clusters);

        // Getting the size of each process's points and summing it up at the end to make sure if numPointsTotal == total_size
        int my_size = 0;
        int total_size = 0;
        for (int i = 0; i < k; ++i)
        {
            my_size = my_size + points_in_clusters.at(i).size();
        }
        MPI_Reduce(&my_size, &total_size, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == Master)
        {
            cout << "Checking the total size after reading the file : " << total_size << endl;
        }

        /*************###DECLARATIONS###*************/
        auto start = std::chrono::high_resolution_clock::now(); // Starting the timer

        char *all_proc_freq_cs;
        char *one_proc_freq_cs;
        int *counts_recv;
        int *displs;

        float E_total = 0.0;
        vector<float> E_history; // To be able to escape an infinite loop where we jump between repeated amounts of E and as a reason never converge.
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
            one_proc_freq_str = one_proc_freq_str + '#'; // adding a # at the end of the string for string-processing

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

            /*************converting the string back to the form of clust_num, att_num , att_val , att_freq ****************/
            /*************to have ALL the data at hand to be able to calculate the Global_mode later**********************/
            /**************** Using a multimap with 2 keys <clust_num,att_num> and a pair of<att_val,freq>****************/
            /*************We need to use a multimap to be able to store more than one value for each key pair*************/

            multimap<pair<int, int>, pair<string, int>> mmp = mmp_builder2(all_proc_freq_cs, all_proc_freq_cs_size);

            /********************FINDING REAL GLOBAL MODES from multimap<pair<int,int>,pair<string , int>> by all the processes********************/
            unordered_map<int, vector<int>> curr_no_of_points;
            unordered_map<int, vector<int>> frequencies;
            unordered_map<int, vector<string>> global_modes = find_mode_global(k, num_att, mmp, frequencies);

            curr_no_points_builder(num_att, points_in_clusters, curr_no_of_points);

            /*There's this condition in which a process doesn't have any more points of a particular cluster. In this case, we put char "%"
            in the place of all the attributes of the unavailable clusters' mode and then ignore these characters in find_mode_partial*/
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

            if (rank == Master)
            {
                cout << "Global modes of clusters in iteration No." << iteration + 1 << " :" << endl;
                for (size_t i = 0; i < global_modes.size(); i++)
                {

                    cout << "--clus-" << to_string(i) << " :" << global_modes[i] << endl;
                }

            } // rank - 0

            /*******RE_CLUSTERING********/
            vector<vector<vector<string>>> points_to_be_added;
            points_to_be_added.resize(k);

            vector<vector<vector<string>>> points_to_be_deleted = re_cluster(points_in_clusters, global_modes, points_to_be_added, frequencies, curr_no_of_points);

            /*Deleting points belongining to other clusters tagged with the special value*/
            vector<string> special_val = {"ts"};
            for (size_t clus_num = 0; clus_num < k; clus_num++)
            {
                points_in_clusters[clus_num].erase(remove(points_in_clusters[clus_num].begin(), points_in_clusters[clus_num].end(), special_val), points_in_clusters[clus_num].end());
            }

            /*Adding points to its right cluster*/
            for (size_t i = 0; i < points_to_be_added.size(); ++i)
            {
                for (size_t j = 0; j < points_to_be_added.at(i).size(); ++j)
                {
                    points_in_clusters[i].push_back(points_to_be_added.at(i).at(j));
                }
            }

            float E_local = E_local_calculator(points_in_clusters, global_modes);
            MPI_Allreduce(&E_local, &E_total, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); // MPI_Allreduce is equal to : MPI_Reduce and then an all-to-all broadcast

            /****************************************************/

            E_history.push_back(E_total);

            // To be able to escape an infinite loop where we jump between repeated amounts of E and as a reason never converge.
            for (size_t i = 0; i < E_history.size() - 1; ++i)
            {
                if (E_total == E_history.at(i))
                {
                    condition = false;
                }
            }
            if (rank == Master)
            {
                cout << "-----------AFTER " << iteration + 1 << " iterations-------------- E===" << E_total << endl;
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
        /********************************To calculate accuracy, precision, and recall *********************************/

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
        char *all_points_in_clusters_cs; // only in Master process
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
            cout << "buf_recv_size = " << all_points_in_clusters_cs_size << endl;

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
            auto time = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
            cout << "after " << time << " ms\n";
            pre_rec_f1_acc_v2(ext_itr + 1, numPointsTotal, all_points_in_clusters, pre_w, rec_w, f1_w, pre_M, rec_M, f1_M, acc);
            itrNum.push_back(iteration);
            timeElapsed.push_back(time);
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
        cout << "After " << total_iterations << " External iteration :\n"
             << endl;
        cout << endl;
        cout << "Ave Elapsed Time   = " << average_calculater(timeElapsed) << " ms" << endl
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
    num_att = numComma; // one-less-att-> + 1;

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

                else
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

// 7:
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

    // unordered_map<int , unordered_map<string , int>> histogram_of_all_atts;//int -> att_num
    unordered_map<int, unordered_map<int, unordered_map<string, int>>> one_proc_freq; // int -> clust_num , int -> att_num

    for (size_t i = 0; i < attributes.size(); i++)
    { // clust_num
        for (size_t j = 0; j < attributes[i].size(); j++)
        { // att_num

            // histogram_of_all_atts[j].push_back(histogram_of_one_att_builder(attributes.at(i).at(j)));
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
        {

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

        if (itr->second > mode_count && itr->first != "%")
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
            { // The function returns an iterator of pairs. The pair refers to the bounds																								range that includes all the elements in the container which have a key equivalent to k

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

void curr_no_points_builder(const int num_att, const vector<vector<vector<string>>> &points_in_clusters, unordered_map<int, vector<int>> &curr_no_of_points)
{ // clust_num

    for (size_t i = 0; i < points_in_clusters.size(); ++i)
    {
        for (size_t j = 0; j < num_att; ++j)
        {
            curr_no_of_points[i].push_back(points_in_clusters[i].size());
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

        // Now that the clust_num and mode_unit_vec are ready, we store them in unordered_map<int,vector<string>> global_modes
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
        {

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
        if (m.count(tp))
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
