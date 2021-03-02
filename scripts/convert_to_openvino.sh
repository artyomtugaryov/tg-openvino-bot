CURRENT_SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"

OUTPUT_PATH="$CURRENT_SCRIPT_PATH/models/ir"


while test $# -gt 0; do
    case "$1" in
        -MODEL)
            MODEL_PATH=$2
            shift 2
            ;;
        -OUT|-O|-o)
            OUTPUT_PATH=$2
            shift 2
            ;;
        *)
            echo -e "Unsupported argument: $1"
            exit 1
            ;;
    esac
done


source ~/intel/openvino_2021/bin/setupvars.sh


python3 ${INSTALLDIR}/deployment_tools/model_optimizer/mo.py --input_model ${MODEL_PATH}  \
                                                             --input_shape [1,256,256,3] \
                                                             --output Squeeze \
                                                             --output_dir ${OUTPUT_PATH} \
                                                             --input "0:F_7/c7s1_32/MirrorPad" \
                                                             --reverse_input_channels


# for horse2zebra and orange2apple --input "0:G_7/c7s1_32/MirrorPad" \