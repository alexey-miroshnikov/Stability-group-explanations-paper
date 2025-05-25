import os, sys
script_dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dirname)
main_dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dirname)


from argparse import ArgumentParser
import json

#########################################################################################
## main
#########################################################################################

def main( ):

    parser = ArgumentParser( description = 'Main script for sability 2 paper.')   
    parser.add_argument('-j', '--json',        
                        default = None,
                        help = '[either full path to json or json name if example_tag is provided]')
    parser.add_argument('-s', '--silent',          
                        action = "store_true", 
                        help = '[either full path to json or json name if example_tag is provided]')
    args = parser.parse_args()

    silent = args.silent

    ###########################################
    # loading json
    ###########################################

    if args.json is None: # (default json)
        filename_json = script_dirname + '/' + "default_credit.json"
    else:
        filename_json = args.json
    assert os.path.exists(filename_json), "{0} does not exist".format(filename_json)

    with open(filename_json, 'r') as f:
        json_dict = json.load(f)

    # pipeline steps:
    pipeline = json_dict["pipeline"]
    step_1_mic_grouping = pipeline["step_1_mic_grouping"]    
    step_2_explain = pipeline["step_2_explain"]	
    step_3_explain_analysis = pipeline["step_3_explain_analysis"]	
    step_4_plot = pipeline["step_4_plot"]


    ###########################################################################################
    # Step 1: variable clustering
    ###########################################################################################

    if step_1_mic_grouping:

        if not silent: 
            print("\n**********************")
            print("[Step 1: variable clustering]")
            print("**********************")

        result = os.system('python ' + script_dirname + '/var_clust.py')

        if result !=0:

            print("\n Step 1 (clustering data) failed!")

    ###########################################################################################
    # Step 2: compute explanations
    ###########################################################################################

    if step_2_explain:

        if not silent: 
            print("\n*******************************")
            print("[Step 2: Computing explanations]")
            print("********************************")

        result = os.system('python ' + script_dirname + '/expl_comp.py')

        if result !=0:

            print("\n Step 2 (explanation computations) failed!")

    ###########################################################################################
    # Step 3: analysis of explanations
    ###########################################################################################

    if step_3_explain_analysis:

        if not silent: 
            print("\n*********************************")
            print("[ Step 3: analysis of explanations ]")
            print("***********************************")

        result = os.system('python ' + script_dirname + '/expl_analysis.py')

        if result !=0:

            print("\n Step 3 (explanation analysis) failed!")

    ###########################################################################################
    # Step 4: plotting
    ###########################################################################################

    if step_4_plot:

        if not silent: 
            print("\n**************************")
            print("[ Step 4: plotting results ]")
            print("**************************")

        result = os.system('python ' + script_dirname + '/plot_results.py')

        if result !=0:

            print("\n Step 4 (plotting results) failed!")

if __name__=="__main__":
        main()

