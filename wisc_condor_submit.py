import os
import sys
import pwd


def main(args):
    '''
    Submit a job using farmoutAnalysisJobs --fwklite
    '''
    print "Begin NN classifying"
    jobName = args.jobName
    sampledir = args.sampledir
    sample_name = os.path.basename(sampledir)
    print "sample_name:", args.samplename
    if sample_name == '':
        print "SAMPLE_NAME not defined, check for trailing '/' on sampledir path"
        return
    else:
        sample_dir = '/nfs_scratch/%s/%s/%s' % (
            pwd.getpwuid(os.getuid())[0], jobName, args.samplename)

    # create submit dir
    submit_dir = '%s/submit' % (sample_dir)
    if os.path.exists(submit_dir):
        print('Submission directory exists for %s %s.' %
              (jobName, args.samplename))

    # create dag dir
    dag_dir = '%s/dags/dag' % (sample_dir)
    os.system('mkdir -p %s' % (os.path.dirname(dag_dir)))
    os.system('mkdir -p %s' % (dag_dir+'inputs'))

    # output dir
    output_dir = 'gsiftp://cms-lvs-gridftp.hep.wisc.edu:2811//hdfs/store/user/%s/%s/%s/'\
        % (pwd.getpwuid(os.getuid())[0], jobName, args.samplename)

    # create file list
    filelist = ['%s/%s' % (sampledir, x) for x in os.listdir(sampledir)]
    filesperjob = 1
    input_name = '%s/%s.txt' % (dag_dir+'inputs', args.samplename)
    with open(input_name, 'w') as file:
        for f in filelist:
            file.write('%s\n' % f.replace('/hdfs', '', 1))

    # create bash script
    bash_name = '%s/%s.sh' % (dag_dir+'inputs', args.samplename)
    bashScript = "#!/bin/bash\n value=$(<$INPUT)\n echo \"$value\"\n"
    bashScript += '$CMSSW_BASE/bin/$SCRAM_ARCH/uniSkim -d %s -j %s -r %s -y %s -l %s -i $value -o \'$OUTPUT\'' % (
        args.samplename, args.job, args.recoil, args.year, args.lepton)
    bashScript += '\n'
    with open(bash_name, 'w') as file:
        file.write(bashScript)
    os.system('chmod +x %s' % bash_name)

    # create farmout command
    farmoutString = 'farmoutAnalysisJobs --infer-cmssw-path --fwklite --input-file-list=%s' % (
        input_name)
    farmoutString += ' --submit-dir=%s --output-dag-file=%s --output-dir=%s' % (
        submit_dir, dag_dir, output_dir)
    farmoutString += ' --input-files-per-job=%i %s %s ' % (
        filesperjob, jobName, bash_name)
    farmoutString += '--use-hdfs'

    if not args.dryrun:
        print('Submitting %s' % args.samplename)
        os.system(farmoutString)
    else:
        print farmoutString

    return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the desired analyzer on FSA n-tuples")
    parser.add_argument('-dr', '--dryrun', action='store_true',
                        help='Create jobs but dont submit')
    parser.add_argument('-j', '--job', action='store', help='job type')
    parser.add_argument('-l', '--lepton', action='store', help='which lepton')
    parser.add_argument('-y', '--year', action='store', help='which year')
    parser.add_argument('-r', '--recoil', action='store', help='recoil type')
    parser.add_argument('-jn', '--jobName', nargs='?', type=str,
                        const='', help='Job Name for condor submission')
    parser.add_argument('-sn', '--samplename', nargs='?',
                        type=str, const='', help='Name of samples')
    parser.add_argument('-sd', '--sampledir', nargs='?',
                        type=str, const='', help='The Sample Input directory')
    args = parser.parse_args()
    main(args)
