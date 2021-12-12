
import scala.io.StdIn.readLine
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.jblas.DoubleMatrix

import breeze.numerics.sqrt
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

class ALSrecommender {

  //推荐
  def recommend(model: MatrixFactorizationModel, movieTitle: Map[Int,String])={
    var choose = ""
    while ( choose != "3"){ //如果选择3离开，纠结束运行程序
      print("请选择要推荐类型 1、针对用户推荐电影 2、针对电影推荐给感兴趣的用户 3、离开？")
      choose = readLine() //读取用户输入
      if(choose == "1"){//如果输入1
        print("请输入用户id:")
        val inputUserID = readLine() //读取用户id
        RecommendMovies(model, movieTitle, inputUserID.toInt) //针对此用户推荐电影
      }else if(choose == "2"){ //如果输入2
        print("请输入电影id:")
        val inputMovieID = readLine() //读取MovieID
        RecommendUsers(model,movieTitle,inputMovieID.toInt) //针对此电影推荐用户
      }
    }
  }
  //针对用户推荐电影程序代码
  def RecommendMovies(model: MatrixFactorizationModel, movieTitle :Map[Int,String], inputUserID: Int) = {
    val RecommendMovie = model.recommendProducts(inputUserID, 10)
    var i = 1
    println("针对用户id-" + inputUserID + "-推荐下列电影：")
    RecommendMovie.foreach{ r =>
      println(i.toString() + "." + movieTitle(r.product) + "评分:" +r.rating.toString())
      i += 1
    }
  }

  //针对电影推荐用户程序代码
  def RecommendUsers(model: MatrixFactorizationModel, movieTitle: Map[Int,String], inputMovieID: Int) = {
    val RecommendUser = model.recommendUsers(inputMovieID, 10) //针对movieID推荐前10位用户
    var i = 1
    println("针对电影id-" + inputMovieID + "-电影名：" + movieTitle(inputMovieID.toInt) + "-推荐下列用户")
    RecommendUser.foreach{ r =>
      println(i.toString() + "用户id:" + r.user + "评分:" + r.rating.toString())
      i = i+1
    }
  }




  //创建数据准备
  def PrepareData(): (RDD[Rating], Map[Int, String]) ={

    //创建用户评分数据
    val sc=new SparkContext(new SparkConf().setAppName("Recommend").setMaster("local[4]"))
    val DataDir="file:/root/workspace/Recommend/data"
    print("开始读取用户评分数据")
    val rawUserData = sc.textFile("file:/root/workspace/Recommend/data/u.data")
    val rawRatings = rawUserData.map(_.split("\t").take(3))
    val ratingsRDD = rawRatings.map { case Array(user, movie, rating) =>Rating(user.toInt, movie.toInt, rating.toDouble)}
    println("共计："+ratingsRDD.count.toString() + "条ratings")

    //创建电影ID与名称对照表
    println("开始读取电影数据中...")
    val itemRDD = sc.textFile(new File(DataDir, "u.item").toString)
    val movieTitle = itemRDD.map(line => line.split("\\|").take(2)).map(array => (array(0).toInt, array(1))).collect().toMap

    //显示数据记录数
    val numRatings = ratingsRDD.count()
    val numUsers = ratingsRDD.map(_.user).distinct().count()
    val numMovies = ratingsRDD.map(_.product).distinct().count()
    println("共计：ratings"+numRatings+" User "+ numUsers + " Movie "+numMovies)
    return (ratingsRDD, movieTitle)
  }



  def main(args:Array[String]){
    SetLogger    //设置不要显示多余信息
    println("======数据准备阶段=====")
    val (ratings,moviesTitle) =PrepareData()
    println("======训练阶段=====")
    println("开始使用"+ratings.count()+"条评比数据进行训练模型")
    val model = ALS.train(ratings, 5, 20,0.1)
    println("======训练完成！=====")
    println("======推荐阶段=====")
    recommend(model, moviesTitle)
    println("======完成！=====")
  }


}
