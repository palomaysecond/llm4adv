"""
@author: xuty
@contact:xuty1218@163.com
@version: 1.0.0
@file: CodeProcess.py
@time: 2025/5/13 15:00
"""
import torch
import os
import time
from datetime import datetime
import argparse
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from model import CodeBERTModel
from run import CodeBertTextDataset, predict_vulnerability, evaluate
from utils import set_seed



def main():
    parser = argparse.ArgumentParser()
    # 基础参数设置
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")  # roberta
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")  # microsoft/codebert-base
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")  # microsoft/codebert-base
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")  # 无值
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")  # saved_models
    parser.add_argument("--test_data_file", default="test.jsonl", type=str) # ？这里参数名字不一样
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")  # 512
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")  # 2
    parser.add_argument("--seed", type=int, default=42)  # 123456
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")  # 无值

    # 添加新的命令行参数
    parser.add_argument("--extract_embeddings", action="store_true",
                        help="Whether to extract and save code embeddings")
    parser.add_argument("--embeddings_file", default="embeddings.npz", type=str,
                        help="Filename to save the extracted embeddings")
    # 指定已微调模型权重路径
    parser.add_argument("--finetuned_model_path", type=str, required=True,
                        help="Path to the fine-tuned model weights (e.g., saved_models/OWASP/checkpoint-best-acc/codebert_model.bin)")
    args = parser.parse_args()

    # args.output_dir = os.path.dirname(args.finetuned_model_path)
    # print(f"[INFO] Output directory automatically set to: {args.output_dir}")
    # 保留命令行传入的 output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Using output directory: {args.output_dir}")

    # 设置设备和随机种子
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # 模型加载
    config_class, model_class, tokenizer_class = {
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
    }[args.model_type]

    # 从 'microsoft/codebert-base' 加载预训练的配置和分词器
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 1  # 表明是一个单标签分类任务
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,do_lower_case=False,
                                                cache_dir=args.cache_dir if args.cache_dir else None)  # 不将文本转为小写
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    model = model_class.from_pretrained(args.model_name_or_path, config=config,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        cache_dir=args.cache_dir if args.cache_dir else None)

    # 将原始模型包装在一个自定义的 Model 类中
    model = CodeBERTModel(model, config, tokenizer, args)

    # 加载已训练的模型权重
    # checkpoint_prefix = 'checkpoint-best-acc/codebert_model.bin'
    # output_dir = os.path.join('saved_models', checkpoint_prefix) # 添加args
    # model.load_state_dict(torch.load(output_dir), strict=False)  # 从最佳精度的检查点加载模型权重，忽略不匹配的键
    if not os.path.isfile(args.finetuned_model_path):
        raise FileNotFoundError(f"Fine-tuned model weights not found at {args.finetuned_model_path}")

    print(f"Loading fine-tuned model weights from: {args.finetuned_model_path}")
    model.load_state_dict(torch.load(args.finetuned_model_path), strict=False)
    model.to(args.device)  # 将模型移动到指定设备上（GPU）

    sample_code = """
        public void processFile(String fileName) {
            BufferedReader reader = null;
            try {
                reader = new BufferedReader(new FileReader(fileName));
                String line;
                while ((line = reader.readLine()) != null) {
                    System.out.println(line);
                }
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                try {
                    if (reader != null) {
                        reader.close();
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        """

    # target=1
    sample_code1 = """
    public class BenchmarkTest01011 extends HttpServlet {
    
        private static final long serialVersionUID = 1L;

        @Override
        public void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
            doPost(request, response);
        }
        
        @Override
        public void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
            
            String param = request.getHeader("foo");
            
            org.owasp.benchmark.helpers.ThingInterface thing = org.owasp.benchmark.helpers.ThingFactory.createThing();
            String bar = thing.doSomething(param);
            
            java.io.File file = new java.io.File(bar);
        }
}
    """

    # target=0
    sample_code2 = """
    public class BenchmarkTest15054 extends HttpServlet {

    private static final long serialVersionUID = 1L;

    @Override
    public void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        doPost(request, response);
    }

    @Override
    public void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {

        String param = request.getHeader("foo");

        String bar = doSomething(param);

        String a1 = "";
        String a2 = "";
        String osName = System.getProperty("os.name");
        if (osName.indexOf("Windows") != -1) {
            a1 = "cmd.exe";
            a2 = "/c";
        } else {
            a1 = "sh";
            a2 = "-c";
        }
        String[] args = {a1, a2, "echo", bar};

        ProcessBuilder pb = new ProcessBuilder();

        pb.command(args);

        try {
            Process p = pb.start();
            org.owasp.benchmark.helpers.Utils.printOSCommandResults(p);
        } catch (IOException e) {
            System.out.println("Problem executing cmdi - java.lang.ProcessBuilder(java.util.List) Test Case");
            throw new ServletException(e);
        }
    }  // end doPost

    private static String doSomething(String param) throws ServletException, IOException {

        java.util.List<String> valuesList = new java.util.ArrayList<String>();
        valuesList.add("safe");
        valuesList.add(param);
        valuesList.add("moresafe");

        valuesList.remove(0); // remove the 1st safe value

        String bar = valuesList.get(1); // get the last 'safe' value

        return bar;
    }
}
    """
    sample_code3 = "public class BenchmarkTest01011 extends HttpServlet {\n\t\n\tprivate static final long serialVersionUID = 1L;\n\t\n\t@Override\n\tpublic void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {\n\t\tdoPost(request, response);\n\t}\n\n\t@Override\n\tpublic void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {\n\t\n\t\tString param = request.getHeader(\"foo\");\n\t\t\n\t\t\n\t\torg.owasp.benchmark.helpers.ThingInterface thing = org.owasp.benchmark.helpers.ThingFactory.createThing();\n\t\tString bar = thing.doSomething(param);\n\t\t\n\t\t\n\t\tjava.io.File file = new java.io.File(bar);\n\t}\n}\n"

    # test
    # is_vulnerable, prob, embeddings, info, e ,l2 ,l2_norm = predict_vulnerability(sample_code3, model, tokenizer, args)
    # print(f"是否有漏洞: {is_vulnerable}, 漏洞概率: {prob:.4f}")
    # print(embeddings)
    # print("-----开始评估-----\n")
    # print(info)
    # print("-----重要性得分-----\n")
    # print(e)
    # print(e.shape)
    # print("\n---L2范数---")
    # print(l2)
    # print(l2.shape)
    # # 统计元素总数
    # total_elements1 = l2.numel()
    # total_elements2 = l2_norm.numel()
    # # 统计非零元素个数
    # non_zero_elements1 = torch.count_nonzero(l2)
    # non_zero_elements2 = torch.count_nonzero(l2_norm)
    #
    # print(f"归一化前元素总数: {total_elements1}\n")
    # print(f"归一化前非零元素个数: {non_zero_elements1.item()}\n")
    # print(f"归一化后元素总数: {total_elements2}\n")
    # print(f"归一化后非零元素个数: {non_zero_elements2.item()}\n")
    #
    # print(l2)
    # print('\n')
    # print(l2_norm)


    # 评估模型，并生成
    result = evaluate(args, model, tokenizer)
    print("Evaluation result:", result)

    code = """
    public class BenchmarkTest01011 extends HttpServlet {
        private static final long serialVersionUID = 1L;

        @Override
        public void doGet(HttpServletRequest request, HttpServletResponse response) {
            doPost(request, response);
        }
    }
    """


if __name__ == '__main__':
    main()